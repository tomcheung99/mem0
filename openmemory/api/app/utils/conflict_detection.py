"""
Real-time conflict detection for new memories.

When a new memory is added, this module checks whether it contradicts
existing active memories for the same user. If a conflict is found,
the system either auto-resolves (archives old, keeps new) or flags both.

Configuration via environment variables:
  CONFLICT_DETECTION_ENABLED – "true" to enable (default: "false")
  CONFLICT_SIMILARITY_THRESHOLD – min cosine score to consider conflict (default: 0.65)
  CONFLICT_ACTION – "auto_resolve" or "flag" (default: "auto_resolve")
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import Memory, MemoryState, MemoryStatusHistory

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

CONFLICT_DETECTION_ENABLED = os.getenv("CONFLICT_DETECTION_ENABLED", "false").lower() == "true"
CONFLICT_SIMILARITY_THRESHOLD = float(os.getenv("CONFLICT_SIMILARITY_THRESHOLD", "0.65"))
CONFLICT_ACTION = os.getenv("CONFLICT_ACTION", "auto_resolve")  # "auto_resolve" | "flag"

CONFLICT_MODEL = os.getenv("CONSOLIDATION_MODEL", "openai/gpt-oss-20b")

# ── LLM prompt ───────────────────────────────────────────────────────────────

CONFLICT_DETECTION_PROMPT = """\
You are a memory conflict detector. You receive an EXISTING memory and a NEW memory \
from the same user.

Determine whether they CONFLICT (cannot both be true at the same time) or are COMPATIBLE.

Guidelines:
- Preference changes count as conflicts: "Likes Python" vs "Prefers Rust now" → CONFLICT
- Factual contradictions are conflicts: "Lives in NYC" vs "Moved to SF" → CONFLICT
- Supplementary info is compatible: "Likes pizza" vs "Likes cheese pizza" → COMPATIBLE
- Unrelated facts are compatible: "Likes pizza" vs "Works as engineer" → COMPATIBLE
- Temporal updates are conflicts: "Is single" vs "Got married" → CONFLICT

Respond ONLY with JSON:
{"verdict": "CONFLICT" | "COMPATIBLE", "conflict_type": "preference_change" | "factual_update" | "correction" | null, "explanation": "brief reason"}
"""


class ConflictVerdict(BaseModel):
    verdict: str  # "CONFLICT" or "COMPATIBLE"
    conflict_type: Optional[str] = None
    explanation: Optional[str] = None


# ── LLM helper ───────────────────────────────────────────────────────────────

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_GATEWAY_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AI_GATEWAY_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required for conflict detection")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _ask_conflict_verdict(existing_content: str, new_content: str) -> ConflictVerdict:
    """Ask LLM whether the new memory conflicts with an existing one."""
    client = _get_openai_client()
    user_msg = (
        f'Existing memory: "{existing_content}"\n'
        f'New memory: "{new_content}"'
    )
    completion = client.beta.chat.completions.parse(
        model=CONFLICT_MODEL,
        messages=[
            {"role": "system", "content": CONFLICT_DETECTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=ConflictVerdict,
        temperature=0,
    )
    return completion.choices[0].message.parsed


# ── Core conflict detection ──────────────────────────────────────────────────

def check_and_resolve_conflicts(
    new_memory: Memory,
    user,
    db: Session,
    memory_client,
    similarity_threshold: float = None,
    action: str = None,
) -> List[Dict[str, Any]]:
    """
    Check a newly added memory against existing active memories for conflicts.

    Returns a list of conflict records (empty if no conflicts).
    Each record: {
        "conflicting_memory_id": str,
        "conflicting_content": str,
        "conflict_type": str,
        "resolution": "auto_resolved" | "flagged",
        "explanation": str,
    }
    """
    if not CONFLICT_DETECTION_ENABLED:
        return []

    threshold = similarity_threshold or CONFLICT_SIMILARITY_THRESHOLD
    resolve_action = action or CONFLICT_ACTION
    conflicts_found: List[Dict[str, Any]] = []

    # 1. Vector search for similar active memories
    candidates = _find_similar_active_memories(new_memory, user, db, memory_client, limit=5)
    if not candidates:
        return []

    # 2. Filter by similarity threshold and ask LLM about conflicts
    for candidate in candidates:
        if candidate["score"] < threshold:
            continue

        try:
            verdict = _ask_conflict_verdict(candidate["content"], new_memory.content)
        except Exception:
            logger.warning("Conflict LLM call failed for memory %s vs %s", candidate["id"], new_memory.id)
            continue

        if verdict.verdict != "CONFLICT":
            continue

        # 3. Resolve conflict
        conflict_record = {
            "conflicting_memory_id": candidate["id"],
            "conflicting_content": candidate["content"],
            "conflict_type": verdict.conflict_type or "unknown",
            "explanation": verdict.explanation or "",
        }

        if resolve_action == "auto_resolve":
            _auto_resolve(candidate["id"], new_memory, user, db, memory_client, verdict)
            conflict_record["resolution"] = "auto_resolved"
        else:
            _flag_conflict(candidate["id"], new_memory, db)
            conflict_record["resolution"] = "flagged"

        conflicts_found.append(conflict_record)

    if conflicts_found:
        db.commit()
        logger.info(
            "Conflict detection for memory %s: %d conflict(s) found",
            new_memory.id,
            len(conflicts_found),
        )

    return conflicts_found


def _find_similar_active_memories(
    new_memory: Memory,
    user,
    db: Session,
    memory_client,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Vector-search for similar active memories (excluding the new memory itself)."""
    try:
        embeddings = memory_client.embedding_model.embed(new_memory.content, "search")
        hits = memory_client.vector_store.search(
            query=new_memory.content,
            vectors=embeddings,
            limit=limit + 5,  # fetch extra to compensate for filtering
            filters={"user_id": user.user_id},
        )
    except Exception:
        logger.warning("Conflict vector search failed for memory %s", new_memory.id)
        return []

    # Only keep active memories that are not the new memory
    active_ids = set(
        str(m.id) for m in db.query(Memory)
        .filter(Memory.user_id == user.id, Memory.state == MemoryState.active)
        .all()
    )

    results = []
    for h in hits:
        hit_id = str(h.id)
        if hit_id == str(new_memory.id):
            continue
        if hit_id not in active_ids:
            continue
        results.append({
            "id": hit_id,
            "content": h.payload.get("data", ""),
            "score": h.score,
        })
        if len(results) >= limit:
            break

    return results


def _auto_resolve(
    old_memory_id: str,
    new_memory: Memory,
    user,
    db: Session,
    memory_client,
    verdict: ConflictVerdict,
) -> None:
    """Archive the old conflicting memory, keep the new one as active."""
    import uuid as _uuid

    old_mem = db.query(Memory).filter(Memory.id == _uuid.UUID(old_memory_id)).first()
    if not old_mem:
        return

    old_mem.state = MemoryState.archived
    old_mem.archived_at = datetime.datetime.now(datetime.timezone.utc)
    old_mem.metadata_ = {
        **(old_mem.metadata_ or {}),
        "conflict_superseded_by": str(new_memory.id),
        "conflict_type": verdict.conflict_type,
        "conflict_explanation": verdict.explanation,
    }

    db.add(MemoryStatusHistory(
        memory_id=old_mem.id,
        changed_by=user.id,
        old_state=MemoryState.active,
        new_state=MemoryState.archived,
    ))

    logger.info(
        "CONFLICT auto-resolved: archived %s (superseded by %s, type=%s)",
        old_memory_id, new_memory.id, verdict.conflict_type,
    )


def _flag_conflict(
    old_memory_id: str,
    new_memory: Memory,
    db: Session,
) -> None:
    """Flag both memories with conflict metadata for manual review."""
    import uuid as _uuid

    # Tag the new memory
    new_memory.metadata_ = {
        **(new_memory.metadata_ or {}),
        "conflict_with": old_memory_id,
        "conflict_resolved": False,
    }

    # Tag the old memory
    old_mem = db.query(Memory).filter(Memory.id == _uuid.UUID(old_memory_id)).first()
    if old_mem:
        old_mem.metadata_ = {
            **(old_mem.metadata_ or {}),
            "conflict_with": str(new_memory.id),
            "conflict_resolved": False,
        }

    logger.info(
        "CONFLICT flagged: %s <-> %s",
        old_memory_id, new_memory.id,
    )
