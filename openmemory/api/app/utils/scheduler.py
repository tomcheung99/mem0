"""
Automatic memory consolidation, decay & dedup scheduler.

Periodically runs:
  1. Memory distillation for all users with pending memories.
  2. Memory decay: archives active memories that haven't been accessed for N days.
  3. Background dedup: pairwise vector similarity check on active memories,
     merging duplicates via the consolidation pipeline.

Configuration via environment variables:
  CONSOLIDATION_ENABLED    – "true" to enable (default: "false")
  CONSOLIDATION_INTERVAL   – seconds between runs (default: 3600 = 1 hour)
  CONSOLIDATION_BATCH_SIZE – max users per cycle (default: 50)
  DECAY_ENABLED            – "true" to enable auto-archive (default: "false")
  DECAY_INTERVAL           – seconds between decay runs (default: 86400 = 24 hours)
  DECAY_IDLE_DAYS          – days without access before archiving (default: 90)
  DEDUP_ENABLED            – "true" to enable background dedup (default: "false")
  DEDUP_INTERVAL           – seconds between dedup runs (default: 43200 = 12 hours)
  DEDUP_SIMILARITY_THRESHOLD – min similarity to trigger merge (default: 0.85)
  DEDUP_BATCH_SIZE         – max users per dedup cycle (default: 20)
"""

import asyncio
import datetime
import logging
import os
from typing import Dict, List, Optional, Set

from app.database import SessionLocal
from app.models import Memory, MemoryState, MemoryStatusHistory, User
from app.utils.memory import get_memory_client
from sqlalchemy import func as sa_func

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

CONSOLIDATION_ENABLED = os.getenv("CONSOLIDATION_ENABLED", "false").lower() == "true"
CONSOLIDATION_INTERVAL = int(os.getenv("CONSOLIDATION_INTERVAL", "3600"))
CONSOLIDATION_BATCH_SIZE = int(os.getenv("CONSOLIDATION_BATCH_SIZE", "50"))

DECAY_ENABLED = os.getenv("DECAY_ENABLED", "false").lower() == "true"
DECAY_INTERVAL = int(os.getenv("DECAY_INTERVAL", "86400"))
DECAY_IDLE_DAYS = int(os.getenv("DECAY_IDLE_DAYS", "90"))

DEDUP_ENABLED = os.getenv("DEDUP_ENABLED", "false").lower() == "true"
DEDUP_INTERVAL = int(os.getenv("DEDUP_INTERVAL", "43200"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.85"))
DEDUP_BATCH_SIZE = int(os.getenv("DEDUP_BATCH_SIZE", "20"))


# ── Background task ──────────────────────────────────────────────────────────

async def _consolidation_loop() -> None:
    """Long-running loop that triggers consolidation at a fixed interval."""
    from app.utils.consolidation import consolidate_user_memories

    logger.info(
        "Consolidation scheduler started (interval=%ds, batch=%d)",
        CONSOLIDATION_INTERVAL,
        CONSOLIDATION_BATCH_SIZE,
    )

    while True:
        await asyncio.sleep(CONSOLIDATION_INTERVAL)
        try:
            await asyncio.to_thread(_run_consolidation_cycle)
        except Exception:
            logger.exception("Consolidation cycle failed")


def _run_consolidation_cycle() -> None:
    """Synchronous: find users with pending memories and consolidate them."""
    from app.utils.consolidation import consolidate_user_memories

    memory_client = get_memory_client()
    if not memory_client:
        logger.warning("Consolidation skipped: memory client unavailable")
        return

    db = SessionLocal()
    try:
        # Find users who have pending memories, limited to batch size
        user_ids_with_pending = (
            db.query(Memory.user_id)
            .filter(Memory.state == MemoryState.pending)
            .distinct()
            .limit(CONSOLIDATION_BATCH_SIZE)
            .all()
        )

        if not user_ids_with_pending:
            logger.debug("Consolidation cycle: no pending memories found")
            return

        logger.info(
            "Consolidation cycle: processing %d user(s)", len(user_ids_with_pending)
        )

        for (uid_uuid,) in user_ids_with_pending:
            user = db.query(User).filter(User.id == uid_uuid).first()
            if not user:
                continue
            try:
                stats = consolidate_user_memories(user.user_id, db, memory_client)
                logger.info(
                    "Consolidated user %s: promoted=%d merged=%d errors=%d",
                    user.user_id,
                    stats.get("promoted", 0),
                    stats.get("merged", 0),
                    stats.get("errors", 0),
                )
            except Exception:
                logger.exception("Consolidation failed for user %s", user.user_id)
    finally:
        db.close()


# ── Decay loop ───────────────────────────────────────────────────────────────

async def _decay_loop() -> None:
    """Long-running loop that archives stale memories at a fixed interval."""
    logger.info(
        "Decay scheduler started (interval=%ds, idle_days=%d)",
        DECAY_INTERVAL,
        DECAY_IDLE_DAYS,
    )

    while True:
        await asyncio.sleep(DECAY_INTERVAL)
        try:
            await asyncio.to_thread(_run_decay_cycle)
        except Exception:
            logger.exception("Decay cycle failed")


def _run_decay_cycle() -> None:
    """Archive active memories that haven't been accessed for DECAY_IDLE_DAYS."""
    db = SessionLocal()
    try:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=DECAY_IDLE_DAYS)

        # Find active memories where:
        #   - last_accessed is set and older than cutoff, OR
        #   - last_accessed is NULL and created_at is older than cutoff (never accessed)
        from sqlalchemy import or_, and_

        stale_memories: List[Memory] = (
            db.query(Memory)
            .filter(
                Memory.state == MemoryState.active,
                or_(
                    and_(Memory.last_accessed != None, Memory.last_accessed < cutoff),  # noqa: E711
                    and_(Memory.last_accessed == None, Memory.created_at < cutoff),  # noqa: E711
                ),
            )
            .limit(500)
            .all()
        )

        if not stale_memories:
            logger.debug("Decay cycle: no stale memories found")
            return

        logger.info("Decay cycle: archiving %d stale memories", len(stale_memories))

        now = datetime.datetime.now(datetime.timezone.utc)
        for mem in stale_memories:
            mem.state = MemoryState.archived
            mem.archived_at = now
            mem.metadata_ = {
                **(mem.metadata_ or {}),
                "archived_reason": "decay",
                "idle_days": DECAY_IDLE_DAYS,
            }
            db.add(MemoryStatusHistory(
                memory_id=mem.id,
                changed_by=mem.user_id,
                old_state=MemoryState.active,
                new_state=MemoryState.archived,
            ))

        db.commit()
        logger.info("Decay cycle: archived %d memories", len(stale_memories))
    except Exception:
        logger.exception("Decay cycle error")
        db.rollback()
    finally:
        db.close()


# ── Dedup loop ───────────────────────────────────────────────────────────────

async def _dedup_loop() -> None:
    """Long-running loop that deduplicates active memories at a fixed interval."""
    logger.info(
        "Dedup scheduler started (interval=%ds, threshold=%.2f, batch=%d)",
        DEDUP_INTERVAL,
        DEDUP_SIMILARITY_THRESHOLD,
        DEDUP_BATCH_SIZE,
    )

    while True:
        await asyncio.sleep(DEDUP_INTERVAL)
        try:
            await asyncio.to_thread(_run_dedup_cycle)
        except Exception:
            logger.exception("Dedup cycle failed")


def _run_dedup_cycle() -> None:
    """Scan active memories per user, merge duplicates above the similarity threshold."""
    from app.utils.consolidation import _ask_merge_or_keep

    memory_client = get_memory_client()
    if not memory_client:
        logger.warning("Dedup skipped: memory client unavailable")
        return

    db = SessionLocal()
    try:
        # Users with at least 2 active memories are worth checking
        user_ids = (
            db.query(Memory.user_id)
            .filter(Memory.state == MemoryState.active)
            .group_by(Memory.user_id)
            .having(sa_func.count(Memory.id) > 1)
            .limit(DEDUP_BATCH_SIZE)
            .all()
        )

        if not user_ids:
            logger.debug("Dedup cycle: no users with >1 active memory")
            return

        total_merged = 0
        for (uid_uuid,) in user_ids:
            user = db.query(User).filter(User.id == uid_uuid).first()
            if not user:
                continue
            try:
                merged = _dedup_user(user, db, memory_client)
                total_merged += merged
            except Exception:
                logger.exception("Dedup failed for user %s", user.user_id)

        if total_merged:
            logger.info("Dedup cycle: merged %d duplicate memories", total_merged)
        else:
            logger.debug("Dedup cycle: no duplicates found")
    finally:
        db.close()


def _dedup_user(user, db, memory_client) -> int:
    """Deduplicate active memories for a single user. Returns count of merges."""
    from app.utils.consolidation import _ask_merge_or_keep
    from app.models import categorize_memory

    active_memories: List[Memory] = (
        db.query(Memory)
        .filter(Memory.user_id == user.id, Memory.state == MemoryState.active)
        .order_by(Memory.created_at)
        .all()
    )

    if len(active_memories) < 2:
        return 0

    merged_ids: Set[str] = set()  # already-consumed memories this cycle
    merge_count = 0

    for mem in active_memories:
        mem_id_str = str(mem.id)
        if mem_id_str in merged_ids:
            continue

        # Vector search for similar active memories
        try:
            embeddings = memory_client.embedding_model.embed(mem.content, "search")
            hits = memory_client.vector_store.search(
                query=mem.content,
                vectors=embeddings,
                limit=10,
                filters={"user_id": user.user_id},
            )
        except Exception:
            logger.warning("Dedup vector search failed for memory %s", mem.id)
            continue

        # Build a set of active IDs for fast lookup
        active_id_set = {str(m.id) for m in active_memories} - merged_ids

        for hit in hits:
            hit_id = str(hit.id)
            # Skip self, already-merged, or non-active
            if hit_id == mem_id_str or hit_id in merged_ids or hit_id not in active_id_set:
                continue

            score = hit.score
            if score < DEDUP_SIMILARITY_THRESHOLD:
                continue

            hit_content = hit.payload.get("data", "")

            # Ask LLM whether these should be merged
            try:
                decision = _ask_merge_or_keep(mem.content, hit_content)
            except Exception:
                logger.warning("Dedup LLM decision failed for %s vs %s", mem.id, hit_id)
                continue

            if decision.action != "MERGE" or not decision.merged_content:
                continue

            # Perform the merge: keep `mem`, consume `hit`
            dup_memory = db.query(Memory).filter(Memory.id == hit_id).first()
            if not dup_memory:
                continue

            # Update keeper content
            _old = mem.content
            mem.content = decision.merged_content
            mem.updated_at = datetime.datetime.now(datetime.timezone.utc)

            # Record version snapshot
            from app.utils.versioning import record_version
            record_version(db, mem, _old, decision.merged_content, "dedup", changed_by=user.id)

            # Re-embed the keeper in vector store
            try:
                memory_client.update(mem_id_str, decision.merged_content)
            except Exception:
                logger.warning("Dedup re-embed failed for %s", mem.id)

            # Delete duplicate from vector store
            try:
                memory_client.vector_store.delete(vector_id=hit_id)
            except Exception:
                logger.warning("Dedup vector delete failed for %s", hit_id)

            # Mark duplicate as merged
            dup_memory.state = MemoryState.merged
            dup_memory.updated_at = datetime.datetime.now(datetime.timezone.utc)
            dup_memory.metadata_ = {
                **(dup_memory.metadata_ or {}),
                "merged_into": mem_id_str,
                "merged_reason": "dedup",
            }
            db.add(MemoryStatusHistory(
                memory_id=dup_memory.id,
                changed_by=user.id,
                old_state=MemoryState.active,
                new_state=MemoryState.merged,
            ))

            merged_ids.add(hit_id)
            merge_count += 1

            logger.info(
                "DEDUP merged %s into %s (score=%.3f)",
                hit_id, mem_id_str, score,
            )

        # Re-categorize the keeper if it was updated
        if merge_count:
            try:
                categorize_memory(mem, db)
            except Exception:
                pass

    if merge_count:
        db.commit()

    return merge_count


# ── Task management ──────────────────────────────────────────────────────────

_scheduler_task: Optional[asyncio.Task] = None
_decay_task: Optional[asyncio.Task] = None
_dedup_task: Optional[asyncio.Task] = None


def start_consolidation_scheduler() -> Optional[asyncio.Task]:
    """Start the background consolidation task if enabled. Call from lifespan."""
    global _scheduler_task
    if not CONSOLIDATION_ENABLED:
        logger.info("Automatic consolidation is disabled (set CONSOLIDATION_ENABLED=true to enable)")
    else:
        _scheduler_task = asyncio.create_task(_consolidation_loop())

    start_decay_scheduler()
    start_dedup_scheduler()
    return _scheduler_task


def start_decay_scheduler() -> Optional[asyncio.Task]:
    """Start the background decay task if enabled."""
    global _decay_task
    if not DECAY_ENABLED:
        logger.info("Memory decay is disabled (set DECAY_ENABLED=true to enable)")
        return None
    _decay_task = asyncio.create_task(_decay_loop())
    return _decay_task


def start_dedup_scheduler() -> Optional[asyncio.Task]:
    """Start the background dedup task if enabled."""
    global _dedup_task
    if not DEDUP_ENABLED:
        logger.info("Background dedup is disabled (set DEDUP_ENABLED=true to enable)")
        return None
    _dedup_task = asyncio.create_task(_dedup_loop())
    return _dedup_task


def stop_consolidation_scheduler() -> None:
    """Cancel all background tasks on shutdown."""
    global _scheduler_task, _decay_task, _dedup_task
    for task_ref in ("_scheduler_task", "_decay_task", "_dedup_task"):
        task = globals().get(task_ref)
        if task and not task.done():
            task.cancel()
    _scheduler_task = None
    _decay_task = None
    _dedup_task = None
