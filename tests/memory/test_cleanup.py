"""Tests for mem0 memory cleanup mechanisms."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytz
import pytest

from mem0.configs.base import (
    CleanupConfig,
    CompactionConfig,
    GarbageCollectionConfig,
    MemoryConfig,
    TemporalDecayConfig,
    TTLConfig,
)
from mem0.memory.cleanup import (
    apply_temporal_decay,
    build_compaction_prompt,
    compute_expires_at,
    is_gc_eligible,
    is_memory_expired,
)


PACIFIC = pytz.timezone("US/Pacific")


# ── Temporal Decay ─────────────────────────────────────────────────

class TestTemporalDecay:
    def test_no_decay_when_just_created(self):
        now_str = datetime.now(PACIFIC).isoformat()
        memories = [
            {"id": "a", "score": 0.9, "created_at": now_str, "updated_at": now_str},
        ]
        result = apply_temporal_decay(memories, decay_rate=0.01)
        # Score should barely change for a memory created now
        assert result[0]["score"] == pytest.approx(0.9, abs=0.01)

    def test_old_memory_decays(self):
        old_ts = (datetime.now(PACIFIC) - timedelta(days=100)).isoformat()
        memories = [
            {"id": "old", "score": 0.9, "created_at": old_ts, "updated_at": old_ts},
        ]
        result = apply_temporal_decay(memories, decay_rate=0.05)
        expected = 0.9 * math.exp(-0.05 * 100)
        assert result[0]["score"] == pytest.approx(expected, rel=0.01)

    def test_sorted_by_adjusted_score(self):
        now = datetime.now(PACIFIC)
        memories = [
            {"id": "old", "score": 0.95, "created_at": (now - timedelta(days=60)).isoformat()},
            {"id": "new", "score": 0.80, "created_at": now.isoformat()},
        ]
        result = apply_temporal_decay(memories, decay_rate=0.02)
        # The newer memory with lower original score should rank higher after decay
        assert result[0]["id"] == "new"
        assert result[1]["id"] == "old"

    def test_fallback_to_created_at(self):
        ts = datetime.now(PACIFIC).isoformat()
        memories = [{"id": "x", "score": 0.5, "created_at": ts}]  # no updated_at
        result = apply_temporal_decay(memories, decay_rate=0.01, time_field="updated_at")
        assert result[0]["score"] == pytest.approx(0.5, abs=0.01)

    def test_missing_timestamp_unchanged(self):
        memories = [{"id": "x", "score": 0.5}]
        result = apply_temporal_decay(memories, decay_rate=0.1)
        assert result[0]["score"] == 0.5


# ── TTL ────────────────────────────────────────────────────────────

class TestTTL:
    def test_not_expired_when_no_expires_at(self):
        assert is_memory_expired({"data": "hello"}) is False

    def test_not_expired_within_ttl(self):
        future = (datetime.now(PACIFIC) + timedelta(hours=1)).isoformat()
        assert is_memory_expired({"expires_at": future}) is False

    def test_expired_past_ttl(self):
        past = (datetime.now(PACIFIC) - timedelta(seconds=1)).isoformat()
        assert is_memory_expired({"expires_at": past}) is True

    def test_compute_expires_at_format(self):
        result = compute_expires_at(3600)
        dt = datetime.fromisoformat(result)
        assert dt > datetime.now(PACIFIC)

    def test_invalid_expires_at_not_expired(self):
        assert is_memory_expired({"expires_at": "not-a-date"}) is False


# ── Compaction Prompt ──────────────────────────────────────────────

class TestCompactionPrompt:
    def test_default_prompt_includes_memories(self):
        memories = [
            {"id": "1", "memory": "Likes coffee"},
            {"id": "2", "memory": "Works remotely"},
        ]
        prompt = build_compaction_prompt(memories)
        assert "Likes coffee" in prompt
        assert "Works remotely" in prompt
        assert "JSON" in prompt

    def test_custom_prompt(self):
        prompt = build_compaction_prompt(
            [{"id": "1", "memory": "test"}],
            custom_prompt="Summarize: {memories}",
        )
        assert prompt.startswith("Summarize:")
        assert "test" in prompt


# ── Garbage Collection ─────────────────────────────────────────────

class TestGarbageCollection:
    def test_recently_accessed_not_eligible(self):
        recent = datetime.now(PACIFIC).isoformat()
        payload = {"last_accessed_at": recent, "access_count": 0}
        assert is_gc_eligible(payload, min_idle_days=7, min_access_count=1) is False

    def test_old_and_unused_is_eligible(self):
        old = (datetime.now(PACIFIC) - timedelta(days=60)).isoformat()
        payload = {"last_accessed_at": old, "access_count": 0}
        assert is_gc_eligible(payload, min_idle_days=30, min_access_count=1) is True

    def test_high_access_count_not_eligible(self):
        old = (datetime.now(PACIFIC) - timedelta(days=60)).isoformat()
        payload = {"last_accessed_at": old, "access_count": 10}
        assert is_gc_eligible(payload, min_idle_days=30, min_access_count=5) is False

    def test_no_timestamps_uses_created_at(self):
        old = (datetime.now(PACIFIC) - timedelta(days=60)).isoformat()
        payload = {"created_at": old, "access_count": 0}
        assert is_gc_eligible(payload, min_idle_days=30, min_access_count=1) is True

    def test_recently_created_not_eligible(self):
        recent = datetime.now(PACIFIC).isoformat()
        payload = {"created_at": recent, "access_count": 0}
        assert is_gc_eligible(payload, min_idle_days=30, min_access_count=1) is False


# ── CleanupConfig ──────────────────────────────────────────────────

class TestCleanupConfig:
    def test_defaults_all_disabled(self):
        cfg = CleanupConfig()
        assert cfg.temporal_decay.enabled is False
        assert cfg.ttl.enabled is False
        assert cfg.compaction.enabled is False
        assert cfg.garbage_collection.enabled is False

    def test_memory_config_includes_cleanup(self):
        mc = MemoryConfig()
        assert hasattr(mc, "cleanup")
        assert isinstance(mc.cleanup, CleanupConfig)

    def test_custom_config(self):
        cfg = CleanupConfig(
            temporal_decay=TemporalDecayConfig(enabled=True, decay_rate=0.05),
            ttl=TTLConfig(enabled=True, default_ttl_seconds=86400),
            compaction=CompactionConfig(enabled=True, max_memories_before_compact=50),
            garbage_collection=GarbageCollectionConfig(enabled=True, min_idle_days=14),
        )
        assert cfg.temporal_decay.decay_rate == 0.05
        assert cfg.ttl.default_ttl_seconds == 86400
        assert cfg.compaction.max_memories_before_compact == 50
        assert cfg.garbage_collection.min_idle_days == 14


# ── Integration: Memory class with cleanup ─────────────────────────

def _make_output_data(id, payload, score=None):
    """Create a mock OutputData-like object."""
    obj = MagicMock()
    obj.id = id
    obj.payload = payload
    obj.score = score
    return obj


class TestMemorySearchWithDecay:
    """Test that temporal decay is applied during search when enabled."""

    @pytest.fixture
    def memory_with_decay(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]) as mock_vs_create, \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                cleanup=CleanupConfig(
                    temporal_decay=TemporalDecayConfig(enabled=True, decay_rate=0.05),
                )
            )
            memory = Memory(config=config)
            yield memory

    def test_decay_applied_to_search_results(self, memory_with_decay):
        old_ts = (datetime.now(PACIFIC) - timedelta(days=30)).isoformat()
        new_ts = datetime.now(PACIFIC).isoformat()

        memory_with_decay.vector_store.search.return_value = [
            _make_output_data("old", {"data": "old fact", "created_at": old_ts}, score=0.95),
            _make_output_data("new", {"data": "new fact", "created_at": new_ts}, score=0.80),
        ]

        results = memory_with_decay._search_vector_store(
            "test query", {"user_id": "u1"}, limit=10
        )

        # Newer memory should rank first due to decay
        assert results[0]["id"] == "new"
        assert results[1]["id"] == "old"
        # Old memory score should be significantly reduced
        assert results[1]["score"] < 0.5


class TestMemorySearchWithTTL:
    """Test that expired memories are filtered out during search."""

    @pytest.fixture
    def memory_with_ttl(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                cleanup=CleanupConfig(
                    ttl=TTLConfig(enabled=True, auto_purge_on_search=True),
                )
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    def test_expired_memories_filtered_from_search(self, memory_with_ttl):
        expired_ts = (datetime.now(PACIFIC) - timedelta(hours=1)).isoformat()
        valid_ts = (datetime.now(PACIFIC) + timedelta(hours=1)).isoformat()

        memory_with_ttl.vector_store.search.return_value = [
            _make_output_data("expired", {"data": "old", "expires_at": expired_ts, "created_at": datetime.now(PACIFIC).isoformat()}, score=0.9),
            _make_output_data("valid", {"data": "still good", "expires_at": valid_ts, "created_at": datetime.now(PACIFIC).isoformat()}, score=0.8),
        ]

        results = memory_with_ttl._search_vector_store(
            "test", {"user_id": "u1"}, limit=10
        )

        assert len(results) == 1
        assert results[0]["id"] == "valid"
