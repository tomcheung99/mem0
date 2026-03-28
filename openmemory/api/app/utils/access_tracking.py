"""
Access tracking for memories.

Bumps `access_count` and `last_accessed` on Memory rows whenever they appear
in search results. This data drives the decay / auto-archive scheduler.
"""

import datetime
import logging
import uuid as _uuid
from typing import Iterable, Union

from sqlalchemy.orm import Session

from app.models import Memory

logger = logging.getLogger(__name__)


def bump_access(
    db: Session,
    memory_ids: Iterable[Union[str, _uuid.UUID]],
    *,
    commit: bool = True,
) -> int:
    """
    Increment access_count and set last_accessed = now for a batch of memory IDs.

    Returns the number of rows updated.
    """
    if not memory_ids:
        return 0

    uuids = []
    for mid in memory_ids:
        try:
            uuids.append(_uuid.UUID(str(mid)))
        except (ValueError, AttributeError):
            continue

    if not uuids:
        return 0

    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        count = (
            db.query(Memory)
            .filter(Memory.id.in_(uuids))
            .update(
                {
                    Memory.access_count: Memory.access_count + 1,
                    Memory.last_accessed: now,
                },
                synchronize_session=False,
            )
        )
        if commit:
            db.commit()
        return count
    except Exception:
        logger.exception("Failed to bump access for %d memories", len(uuids))
        db.rollback()
        return 0
