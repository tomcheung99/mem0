"""
Memory content versioning – records an immutable snapshot every time a
memory's content is changed (update, merge, dedup, etc.).

Usage:
    from app.utils.versioning import record_version
    record_version(db, memory, old_content, new_content, "update", changed_by=user.id)
"""

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from app.models import MemoryVersion

logger = logging.getLogger(__name__)


def record_version(
    db: Session,
    memory,
    old_content: str,
    new_content: str,
    change_type: str,
    changed_by: Optional[UUID] = None,
) -> Optional[MemoryVersion]:
    """Create a MemoryVersion row. Skips silently if content hasn't changed."""
    if old_content == new_content:
        return None

    # Determine next version number
    max_ver = (
        db.query(sa_func.max(MemoryVersion.version))
        .filter(MemoryVersion.memory_id == memory.id)
        .scalar()
    )
    next_ver = (max_ver or 0) + 1

    ver = MemoryVersion(
        memory_id=memory.id,
        version=next_ver,
        old_content=old_content,
        new_content=new_content,
        change_type=change_type,
        changed_by=changed_by,
    )
    db.add(ver)
    logger.debug(
        "Recorded version %d for memory %s (type=%s)", next_ver, memory.id, change_type
    )
    return ver
