"""
Memory cleanup mechanisms for mem0.

Provides four core strategies:
1. Temporal Decay   – Score = Relevance × e^(-λt), applied at search time.
2. Event-driven TTL – Memories expire after a configurable TTL.
3. Compaction       – Summarise many fine-grained memories into dense blocks via LLM.
4. Garbage Collection – Periodic removal of unused / low-score memories.
5. Proactive Forgetting – Entropy tracking and tier-based TTL application.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytz

logger = logging.getLogger(__name__)

# ── helpers ─────────────────────────────────────────────────────────
PACIFIC = pytz.timezone("US/Pacific")

_DEFAULT_COMPACTION_PROMPT = (
    "You are a knowledge curator. You will receive a list of individual memory entries that "
    "belong to the same user / agent session. Condense them into a compact set of high-density "
    "knowledge statements. Each statement should be self-contained. Remove redundancy and merge "
    'overlapping facts. Output ONLY a JSON object: {{"summaries": ["statement1", "statement2", ...]}}\n\n'
    "Memory entries:\n{memories}"
)


def _parse_iso_to_utc(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string to a timezone-aware UTC datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = PACIFIC.localize(dt)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── 1.  Temporal Decay ─────────────────────────────────────────────

def apply_temporal_decay(
    memories: List[Dict[str, Any]],
    decay_rate: float,
    time_field: str = "updated_at",
) -> List[Dict[str, Any]]:
    """
    Adjust each memory's ``score`` with exponential time decay.

    Formula: adjusted_score = original_score × e^(-λ × Δdays)

    Args:
        memories: List of memory dicts (must contain ``score`` and timestamp fields).
        decay_rate: λ - the decay coefficient per day.
        time_field: Which field to use for the age calculation.
                    Falls back to ``created_at`` when the preferred field is absent.

    Returns:
        The same list, mutated in-place and re-sorted by the new ``score``.
    """
    now = _now_utc()

    for mem in memories:
        ts_str = mem.get(time_field) or mem.get("created_at")
        ts = _parse_iso_to_utc(ts_str)
        if ts is None:
            continue

        delta_days = max((now - ts).total_seconds() / 86400.0, 0.0)
        original_score = mem.get("score") or 0.0
        mem["score"] = original_score * math.exp(-decay_rate * delta_days)

    memories.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return memories


# ── 2.  TTL helpers ─────────────────────────────────────────────────

def is_memory_expired(payload: Dict[str, Any]) -> bool:
    """
    Check whether a memory has passed its TTL.

    A memory is expired when *both* conditions are true:
    * ``expires_at`` is present in the payload.
    * The current time is past ``expires_at``.
    """
    expires_str = payload.get("expires_at")
    if not expires_str:
        return False
    expires_at = _parse_iso_to_utc(expires_str)
    if expires_at is None:
        return False
    return _now_utc() > expires_at


def compute_expires_at(ttl_seconds: int) -> str:
    """Return an ISO-8601 ``expires_at`` timestamp *ttl_seconds* from now (Pacific TZ)."""
    return (datetime.now(PACIFIC) + timedelta(seconds=ttl_seconds)).isoformat()


# ── 3.  Compaction ──────────────────────────────────────────────────

def build_compaction_prompt(
    memories: List[Dict[str, Any]],
    custom_prompt: Optional[str] = None,
) -> str:
    """
    Build the LLM prompt used to condense a list of memories.

    Args:
        memories: Dicts that must contain at least an ``id`` and ``memory`` key.
        custom_prompt: Optional override prompt. Must contain ``{memories}`` placeholder.

    Returns:
        The fully formatted prompt string ready for LLM consumption.
    """
    template = custom_prompt or _DEFAULT_COMPACTION_PROMPT
    memory_lines = "\n".join(
        f"- [{m.get('id', '?')}] {m.get('memory', '')}" for m in memories
    )
    return template.format(memories=memory_lines)


# ── 4.  Garbage Collection scoring ─────────────────────────────────

def is_gc_eligible(
    payload: Dict[str, Any],
    min_idle_days: float,
    min_access_count: int,
) -> bool:
    """
    Determine whether a memory qualifies for garbage collection.

    A memory is GC-eligible when *all* of the following hold:
    * It has not been accessed for at least ``min_idle_days``.
    * Its ``access_count`` is below ``min_access_count``.

    The caller may additionally check the decayed score against a threshold;
    this function focuses on access-pattern criteria only.
    """
    now = _now_utc()

    last_accessed_str = payload.get("last_accessed_at")
    if last_accessed_str:
        last_accessed = _parse_iso_to_utc(last_accessed_str)
        if last_accessed and (now - last_accessed).total_seconds() / 86400.0 < min_idle_days:
            return False
    else:
        # If never accessed, use created_at as the baseline
        created_str = payload.get("created_at")
        created = _parse_iso_to_utc(created_str)
        if created and (now - created).total_seconds() / 86400.0 < min_idle_days:
            return False

    access_count = payload.get("access_count", 0)
    if access_count >= min_access_count:
        return False

    return True


# ── 5.  Proactive Forgetting ──────────────────────────────────────

def compute_memory_entropy(
    memories: List[Dict[str, Any]],
    window_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    Compute the "entropy" (noise ratio) of recently created memories.

    Entropy is measured as the ratio of low-trust memories to total memories
    within the time window. A high entropy value indicates many low-quality
    memories are being accumulated (noise).

    Args:
        memories: List of memory dicts with ``trust_score`` and timestamp fields.
        window_hours: Only consider memories created within this window.

    Returns:
        Dict with ``entropy`` (0.0-1.0), ``total_in_window``, ``low_trust_count``,
        and ``avg_trust_score``.
    """
    now = _now_utc()
    cutoff = now - timedelta(hours=window_hours)

    in_window = []
    for mem in memories:
        ts_str = mem.get("created_at") or mem.get("updated_at")
        ts = _parse_iso_to_utc(ts_str)
        if ts and ts >= cutoff:
            in_window.append(mem)

    if not in_window:
        return {"entropy": 0.0, "total_in_window": 0, "low_trust_count": 0, "avg_trust_score": 0.0}

    trust_scores = [m.get("trust_score", 0.5) for m in in_window]
    avg_score = sum(trust_scores) / len(trust_scores)
    low_trust_count = sum(1 for s in trust_scores if s < 0.4)
    entropy = low_trust_count / len(in_window)

    return {
        "entropy": round(entropy, 4),
        "total_in_window": len(in_window),
        "low_trust_count": low_trust_count,
        "avg_trust_score": round(avg_score, 4),
    }


def compute_tier_ttl(
    memory_tier: str,
    tier_ttl_map: Dict[str, Optional[int]],
) -> Optional[str]:
    """
    Compute an ``expires_at`` timestamp based on the memory's tier.

    Args:
        memory_tier: One of "working", "session", "long_term", "archived".
        tier_ttl_map: Mapping from tier name to TTL in seconds. None = no expiration.

    Returns:
        ISO-8601 ``expires_at`` string, or None if the tier has no TTL.
    """
    ttl_seconds = tier_ttl_map.get(memory_tier)
    if ttl_seconds is None:
        return None
    return compute_expires_at(ttl_seconds)


def apply_temporal_trust_boost(
    trust_score: float,
    created_at_str: Optional[str],
    temporal_weight_factor: float = 0.1,
    max_boost: float = 0.15,
) -> float:
    """
    Boost a memory's trust score based on recency.

    Newer memories receive a small boost to their trust score,
    reflecting the principle that recent information is more likely
    to be currently relevant.

    Args:
        trust_score: Original trust score (0.0-1.0).
        created_at_str: ISO timestamp of when the memory was created.
        temporal_weight_factor: Factor controlling boost magnitude per day of recency.
        max_boost: Maximum boost that can be applied.

    Returns:
        Adjusted trust score, capped at 1.0.
    """
    if not created_at_str:
        return trust_score

    created_at = _parse_iso_to_utc(created_at_str)
    if not created_at:
        return trust_score

    now = _now_utc()
    days_old = max((now - created_at).total_seconds() / 86400.0, 0.0)
    # Boost decays with age: boost = max_boost * e^(-factor * days)
    boost = max_boost * math.exp(-temporal_weight_factor * days_old)
    return min(trust_score + boost, 1.0)
