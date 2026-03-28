import datetime

from app.database import get_db
from app.models import App, Memory, MemoryState, User
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1/stats", tags=["stats"])

@router.get("/")
async def get_profile(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get total number of memories
    total_memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted).count()

    # Get total number of apps
    apps = db.query(App).filter(App.owner == user)
    total_apps = apps.count()

    return {
        "total_memories": total_memories,
        "total_apps": total_apps,
        "apps": apps.all()
    }


@router.get("/memory-health/{user_id}")
async def get_memory_health(
    user_id: str,
    db: Session = Depends(get_db),
):
    """
    Memory health dashboard for a single user.

    Returns counts by state, conflict flags, age stats, access stats,
    and a lightweight duplicate-risk heuristic.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    uid = user.id

    # ── Per-state counts ─────────────────────────────────────────────────
    state_counts = dict(
        db.query(Memory.state, sa_func.count(Memory.id))
        .filter(Memory.user_id == uid, Memory.state != MemoryState.deleted)
        .group_by(Memory.state)
        .all()
    )

    active = state_counts.get(MemoryState.active, 0)
    pending = state_counts.get(MemoryState.pending, 0)
    archived = state_counts.get(MemoryState.archived, 0)
    merged = state_counts.get(MemoryState.merged, 0)
    paused = state_counts.get(MemoryState.paused, 0)

    # ── Flagged conflicts (metadata contains "conflict" key) ─────────────
    flagged_conflicts = (
        db.query(sa_func.count(Memory.id))
        .filter(
            Memory.user_id == uid,
            Memory.state == MemoryState.active,
            Memory.metadata_.op("->")("conflict").isnot(None),
        )
        .scalar()
    ) or 0

    # ── Oldest memory age in days ────────────────────────────────────────
    oldest_created = (
        db.query(sa_func.min(Memory.created_at))
        .filter(Memory.user_id == uid, Memory.state != MemoryState.deleted)
        .scalar()
    )
    if oldest_created:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if oldest_created.tzinfo is None:
            oldest_created = oldest_created.replace(tzinfo=datetime.timezone.utc)
        oldest_memory_days = (now_utc - oldest_created).days
    else:
        oldest_memory_days = 0

    # ── Average access count ─────────────────────────────────────────────
    avg_access = (
        db.query(sa_func.avg(Memory.access_count))
        .filter(Memory.user_id == uid, Memory.state == MemoryState.active)
        .scalar()
    )
    avg_access_count = round(float(avg_access), 2) if avg_access else 0.0

    # ── Duplicate risk score (heuristic) ─────────────────────────────────
    # Ratio of merged memories to total non-deleted memories gives a rough
    # estimate of how duplicate-prone the user's memory set is.
    total_non_deleted = active + pending + archived + merged + paused
    if total_non_deleted > 0:
        duplicate_risk_score = round(merged / total_non_deleted, 2)
    else:
        duplicate_risk_score = 0.0

    return {
        "active": active,
        "pending": pending,
        "archived": archived,
        "merged": merged,
        "paused": paused,
        "flagged_conflicts": flagged_conflicts,
        "oldest_memory_days": oldest_memory_days,
        "avg_access_count": avg_access_count,
        "duplicate_risk_score": duplicate_risk_score,
    }

