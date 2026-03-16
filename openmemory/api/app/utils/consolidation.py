"""
Candidate → Canonical memory consolidation pipeline.

Pending memories (candidates) are promoted or merged into canonical (active) memories:
  - PROMOTE: candidate is a genuinely new fact → state becomes active
  - MERGE:   candidate overlaps with an existing active memory → content merged,
             candidate marked as merged, canonical re-embedded
"""

import datetime
import json
import logging
import os
import uuid
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import (
    Category,
    Memory,
    MemoryState,
    MemoryStatusHistory,
    categorize_memory,
    memory_categories,
)

logger = logging.getLogger(__name__)

# ── LLM helpers ──────────────────────────────────────────────────────────────

CONSOLIDATION_PROMPT = """\
You are a memory consolidation system. You receive ONE existing canonical memory \
and ONE new candidate memory from the same user.

Decide:
• MERGE  – the candidate overlaps with or extends the existing memory. \
  Produce a single consolidated sentence that preserves ALL useful details from both.
• KEEP_BOTH – the candidate is clearly about a different topic or adds a distinct fact.

Rules:
- Be concise. One or two sentences max for merged_content.
- Never lose information present in either memory.
- If the candidate is simply a rephrasing of the existing memory with no new info, MERGE and keep the better phrasing.
- When in doubt, KEEP_BOTH.

Respond ONLY with JSON:
{"action": "MERGE" | "KEEP_BOTH", "merged_content": "..." }
(merged_content is required only when action is MERGE)
"""


CONSOLIDATION_MODEL = os.getenv("CONSOLIDATION_MODEL", "openai/gpt-oss-20b")


class ConsolidationDecision(BaseModel):
    action: str  # "MERGE" or "KEEP_BOTH"
    merged_content: Optional[str] = None


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_GATEWAY_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AI_GATEWAY_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AI_GATEWAY_API_KEY required for consolidation")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _ask_merge_or_keep(existing_content: str, candidate_content: str) -> ConsolidationDecision:
    """Ask LLM whether to merge candidate into existing or keep both."""
    client = _get_openai_client()
    user_msg = (
        f'Existing memory: "{existing_content}"\n'
        f'New candidate: "{candidate_content}"'
    )
    completion = client.beta.chat.completions.parse(
        model=CONSOLIDATION_MODEL,
        messages=[
            {"role": "system", "content": CONSOLIDATION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=ConsolidationDecision,
        temperature=0,
    )
    return completion.choices[0].message.parsed


# ── Core consolidation ───────────────────────────────────────────────────────

def consolidate_user_memories(
    user_id: str,
    db: Session,
    memory_client,
    similarity_threshold: float = 0.6,
) -> Dict:
    """
    Consolidate all pending (candidate) memories for a user.

    Pipeline per candidate:
      1. Vector-search active memories for the closest match.
      2. If score >= threshold, ask LLM → MERGE or KEEP_BOTH.
      3. MERGE  → update canonical content + re-embed, mark candidate as merged.
      4. KEEP_BOTH / no match → promote candidate to active.
      5. Categorize every newly promoted or merged-into memory.

    Returns a stats dict: {promoted, merged, errors, total}.
    """
    from app.models import Memory, MemoryState, MemoryStatusHistory

    stats = {"promoted": 0, "merged": 0, "errors": 0, "total": 0}

    from app.models import User
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        logger.warning("consolidate: user %s not found", user_id)
        return stats

    pending_memories = (
        db.query(Memory)
        .filter(Memory.user_id == user.id, Memory.state == MemoryState.pending)
        .order_by(Memory.created_at)
        .all()
    )
    stats["total"] = len(pending_memories)
    if not pending_memories:
        logger.info("consolidate: no pending memories for user %s", user_id)
        return stats

    logger.info("consolidate: %d pending memories for user %s", len(pending_memories), user_id)

    for candidate in pending_memories:
        try:
            _consolidate_one(candidate, user, db, memory_client, similarity_threshold, stats)
        except Exception:
            logger.exception("consolidate: error processing memory %s", candidate.id)
            stats["errors"] += 1

    db.commit()
    return stats


def _consolidate_one(
    candidate: Memory,
    user,
    db: Session,
    memory_client,
    similarity_threshold: float,
    stats: Dict,
) -> None:
    """Process a single pending candidate."""

    # 1. Vector search for similar active memories
    best_match = _find_best_active_match(candidate, user, db, memory_client)

    should_merge = False
    if best_match and best_match["score"] >= similarity_threshold:
        # 2. Ask LLM whether to merge
        try:
            decision = _ask_merge_or_keep(best_match["content"], candidate.content)
            should_merge = decision.action == "MERGE" and decision.merged_content
        except Exception:
            logger.warning("LLM merge decision failed for %s, will promote", candidate.id)

    if should_merge:
        _do_merge(candidate, best_match, decision.merged_content, user, db, memory_client)
        stats["merged"] += 1
    else:
        _do_promote(candidate, user, db)
        stats["promoted"] += 1


def _find_best_active_match(
    candidate: Memory,
    user,
    db: Session,
    memory_client,
) -> Optional[Dict]:
    """Vector-search for the most similar active memory."""
    try:
        embeddings = memory_client.embedding_model.embed(candidate.content, "search")
        hits = memory_client.vector_store.search(
            query=candidate.content,
            vectors=embeddings,
            limit=10,
            filters={"user_id": user.user_id},
        )
    except Exception:
        logger.warning("Vector search failed for candidate %s", candidate.id)
        return None

    # Filter to only active memories (not the candidate itself)
    active_ids = set(
        str(m.id) for m in db.query(Memory)
        .filter(Memory.user_id == user.id, Memory.state == MemoryState.active)
        .all()
    )

    best = None
    for h in hits:
        hit_id = str(h.id)
        if hit_id == str(candidate.id):
            continue
        if hit_id not in active_ids:
            continue

        score = h.score
        content = h.payload.get("data", "")

        # Apply reranker if available
        if hasattr(memory_client, "reranker") and memory_client.reranker:
            try:
                reranked = memory_client.reranker.rerank(
                    candidate.content,
                    [{"memory": content, "id": hit_id}],
                    top_k=1,
                )
                if reranked:
                    score = reranked[0].get("rerank_score", score)
            except Exception:
                pass

        if best is None or score > best["score"]:
            best = {"id": hit_id, "content": content, "score": score}

    return best


def _do_merge(
    candidate: Memory,
    active_match: Dict,
    merged_content: str,
    user,
    db: Session,
    memory_client,
) -> None:
    """Merge candidate into the existing active memory."""
    active_id = uuid.UUID(active_match["id"])
    active_memory = db.query(Memory).filter(Memory.id == active_id).first()
    if not active_memory:
        # Fallback: promote instead
        _do_promote(candidate, user, db)
        return

    old_content = active_memory.content

    # Update canonical memory content in SQL
    active_memory.content = merged_content
    active_memory.updated_at = datetime.datetime.now(datetime.UTC)

    # Re-embed in vector store
    try:
        memory_client.update(str(active_id), merged_content)
    except Exception:
        logger.warning("Vector re-embed failed for %s, SQL updated anyway", active_id)

    # Delete candidate from vector store (mem0 already wrote it there on add)
    try:
        memory_client.vector_store.delete(vector_id=str(candidate.id))
    except Exception:
        logger.warning("Vector delete of candidate %s failed", candidate.id)

    # Mark candidate as merged
    candidate.state = MemoryState.merged
    candidate.updated_at = datetime.datetime.now(datetime.UTC)
    candidate.metadata_ = {
        **(candidate.metadata_ or {}),
        "merged_into": str(active_id),
    }

    # History for candidate
    db.add(MemoryStatusHistory(
        memory_id=candidate.id,
        changed_by=user.id,
        old_state=MemoryState.pending,
        new_state=MemoryState.merged,
    ))

    # Categorize the updated canonical memory
    try:
        categorize_memory(active_memory, db)
    except Exception:
        logger.warning("Categorization failed for merged memory %s", active_id)

    logger.info(
        "MERGED candidate %s into active %s (old: %r → new: %r)",
        candidate.id, active_id, old_content[:60], merged_content[:60],
    )


def _do_promote(candidate: Memory, user, db: Session) -> None:
    """Promote a pending candidate to active."""
    candidate.state = MemoryState.active
    candidate.updated_at = datetime.datetime.now(datetime.UTC)

    db.add(MemoryStatusHistory(
        memory_id=candidate.id,
        changed_by=user.id,
        old_state=MemoryState.pending,
        new_state=MemoryState.active,
    ))

    # Categorize the promoted memory
    try:
        categorize_memory(candidate, db)
    except Exception:
        logger.warning("Categorization failed for promoted memory %s", candidate.id)

    logger.info("PROMOTED candidate %s to active", candidate.id)
