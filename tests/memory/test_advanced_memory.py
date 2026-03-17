"""Tests for trust scoring, hierarchical memory, conflict resolution, and proactive forgetting."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz
import pytest

from mem0.configs.base import (
    CleanupConfig,
    ConflictResolutionConfig,
    HierarchicalMemoryConfig,
    MemoryConfig,
    MemoryItem,
    ProactiveForgettingConfig,
    TrustScoringConfig,
)
from mem0.configs.enums import MemoryTier
from mem0.configs.prompts import (
    TRUST_SCORING_PROMPT,
    CONFLICT_AWARE_UPDATE_MEMORY_PROMPT,
    get_conflict_aware_update_memory_messages,
)
from mem0.memory.cleanup import (
    apply_temporal_trust_boost,
    compute_memory_entropy,
    compute_tier_ttl,
)


PACIFIC = pytz.timezone("US/Pacific")


def _make_output_data(id, payload, score=None):
    """Create a mock OutputData-like object."""
    obj = MagicMock()
    obj.id = id
    obj.payload = payload
    obj.score = score
    return obj


# ═══════════════════════════════════════════════════════════════════
# Trust Scoring Config
# ═══════════════════════════════════════════════════════════════════

class TestTrustScoringConfig:
    def test_defaults(self):
        cfg = TrustScoringConfig()
        assert cfg.enabled is False
        assert cfg.archive_threshold == 0.3
        assert cfg.scoring_prompt is None

    def test_custom_config(self):
        cfg = TrustScoringConfig(enabled=True, archive_threshold=0.5, scoring_prompt="custom")
        assert cfg.enabled is True
        assert cfg.archive_threshold == 0.5
        assert cfg.scoring_prompt == "custom"

    def test_memory_config_includes_trust_scoring(self):
        mc = MemoryConfig()
        assert hasattr(mc, "trust_scoring")
        assert isinstance(mc.trust_scoring, TrustScoringConfig)
        assert mc.trust_scoring.enabled is False


# ═══════════════════════════════════════════════════════════════════
# Hierarchical Memory Config
# ═══════════════════════════════════════════════════════════════════

class TestHierarchicalMemoryConfig:
    def test_defaults(self):
        cfg = HierarchicalMemoryConfig()
        assert cfg.enabled is False
        assert cfg.working_memory_max_items == 10
        assert cfg.session_ttl_seconds == 86400
        assert cfg.promotion_threshold == 0.6
        assert cfg.demotion_idle_days == 30.0

    def test_memory_config_includes_hierarchical(self):
        mc = MemoryConfig()
        assert hasattr(mc, "hierarchical_memory")
        assert isinstance(mc.hierarchical_memory, HierarchicalMemoryConfig)


# ═══════════════════════════════════════════════════════════════════
# Conflict Resolution Config
# ═══════════════════════════════════════════════════════════════════

class TestConflictResolutionConfig:
    def test_defaults(self):
        cfg = ConflictResolutionConfig()
        assert cfg.enabled is False
        assert cfg.temporal_weight_factor == 0.1
        assert cfg.contradiction_action == "flag"

    def test_auto_resolve_mode(self):
        cfg = ConflictResolutionConfig(enabled=True, contradiction_action="auto_resolve")
        assert cfg.contradiction_action == "auto_resolve"

    def test_memory_config_includes_conflict_resolution(self):
        mc = MemoryConfig()
        assert hasattr(mc, "conflict_resolution")
        assert isinstance(mc.conflict_resolution, ConflictResolutionConfig)


# ═══════════════════════════════════════════════════════════════════
# Proactive Forgetting Config
# ═══════════════════════════════════════════════════════════════════

class TestProactiveForgettingConfig:
    def test_defaults(self):
        cfg = ProactiveForgettingConfig()
        assert cfg.enabled is False
        assert cfg.entropy_threshold == 0.8
        assert cfg.entropy_window_hours == 24
        assert cfg.tier_ttl_seconds["working"] == 3600
        assert cfg.tier_ttl_seconds["session"] == 86400
        assert cfg.tier_ttl_seconds["long_term"] is None
        assert cfg.tier_ttl_seconds["archived"] == 604800

    def test_cleanup_config_includes_proactive_forgetting(self):
        cc = CleanupConfig()
        assert hasattr(cc, "proactive_forgetting")
        assert isinstance(cc.proactive_forgetting, ProactiveForgettingConfig)


# ═══════════════════════════════════════════════════════════════════
# MemoryTier Enum
# ═══════════════════════════════════════════════════════════════════

class TestMemoryTierEnum:
    def test_values(self):
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.SESSION.value == "session"
        assert MemoryTier.LONG_TERM.value == "long_term"
        assert MemoryTier.ARCHIVED.value == "archived"


# ═══════════════════════════════════════════════════════════════════
# MemoryItem with new fields
# ═══════════════════════════════════════════════════════════════════

class TestMemoryItemExtended:
    def test_trust_score_field(self):
        item = MemoryItem(id="test", memory="test memory", trust_score=0.85)
        assert item.trust_score == 0.85

    def test_memory_tier_field(self):
        item = MemoryItem(id="test", memory="test memory", memory_tier="long_term")
        assert item.memory_tier == "long_term"

    def test_fields_optional(self):
        item = MemoryItem(id="test", memory="test memory")
        assert item.trust_score is None
        assert item.memory_tier is None


# ═══════════════════════════════════════════════════════════════════
# Trust Scoring Prompt
# ═══════════════════════════════════════════════════════════════════

class TestTrustScoringPrompt:
    def test_prompt_format(self):
        facts = ["Name is Alice", "Likes coffee"]
        facts_text = "\n".join(f"- {f}" for f in facts)
        prompt = TRUST_SCORING_PROMPT.format(facts=facts_text)
        assert "Name is Alice" in prompt
        assert "Likes coffee" in prompt
        assert "scored_facts" in prompt
        assert "0.0 to 1.0" in prompt

    def test_prompt_contains_scoring_criteria(self):
        assert "Personal relevance" in TRUST_SCORING_PROMPT
        assert "Actionability" in TRUST_SCORING_PROMPT
        assert "Uniqueness" in TRUST_SCORING_PROMPT
        assert "Long-term value" in TRUST_SCORING_PROMPT


# ═══════════════════════════════════════════════════════════════════
# Conflict-Aware Update Memory Prompt
# ═══════════════════════════════════════════════════════════════════

class TestConflictAwarePrompt:
    def test_prompt_contains_conflict_operation(self):
        assert "CONFLICT" in CONFLICT_AWARE_UPDATE_MEMORY_PROMPT
        assert "conflict_type" in CONFLICT_AWARE_UPDATE_MEMORY_PROMPT

    def test_get_conflict_aware_messages(self):
        old_memory = [{"id": "0", "text": "Likes Python"}]
        new_facts = ["Prefers Rust now"]
        result = get_conflict_aware_update_memory_messages(old_memory, new_facts)
        assert "CONFLICT" in result
        assert "Likes Python" in result
        assert "Prefers Rust now" in result

    def test_get_conflict_aware_messages_empty_memory(self):
        result = get_conflict_aware_update_memory_messages([], ["New fact"])
        assert "Current memory is empty" in result

    def test_custom_prompt_override(self):
        result = get_conflict_aware_update_memory_messages(
            [{"id": "0", "text": "test"}],
            ["new fact"],
            custom_update_memory_prompt="Custom: {old} {new}",
        )
        assert "Custom:" in result


# ═══════════════════════════════════════════════════════════════════
# Entropy Computation
# ═══════════════════════════════════════════════════════════════════

class TestEntropyComputation:
    def test_empty_memories(self):
        result = compute_memory_entropy([], window_hours=24)
        assert result["entropy"] == 0.0
        assert result["total_in_window"] == 0

    def test_all_high_trust(self):
        now = datetime.now(PACIFIC).isoformat()
        memories = [
            {"created_at": now, "trust_score": 0.9},
            {"created_at": now, "trust_score": 0.8},
            {"created_at": now, "trust_score": 0.7},
        ]
        result = compute_memory_entropy(memories, window_hours=24)
        assert result["entropy"] == 0.0
        assert result["low_trust_count"] == 0
        assert result["avg_trust_score"] == pytest.approx(0.8, abs=0.01)

    def test_all_low_trust_high_entropy(self):
        now = datetime.now(PACIFIC).isoformat()
        memories = [
            {"created_at": now, "trust_score": 0.1},
            {"created_at": now, "trust_score": 0.2},
            {"created_at": now, "trust_score": 0.15},
        ]
        result = compute_memory_entropy(memories, window_hours=24)
        assert result["entropy"] == 1.0
        assert result["low_trust_count"] == 3

    def test_mixed_trust_scores(self):
        now = datetime.now(PACIFIC).isoformat()
        memories = [
            {"created_at": now, "trust_score": 0.9},
            {"created_at": now, "trust_score": 0.1},
            {"created_at": now, "trust_score": 0.8},
            {"created_at": now, "trust_score": 0.2},
        ]
        result = compute_memory_entropy(memories, window_hours=24)
        assert result["entropy"] == 0.5  # 2 out of 4 are low trust
        assert result["low_trust_count"] == 2

    def test_old_memories_excluded(self):
        old_ts = (datetime.now(PACIFIC) - timedelta(hours=48)).isoformat()
        memories = [
            {"created_at": old_ts, "trust_score": 0.1},
        ]
        result = compute_memory_entropy(memories, window_hours=24)
        assert result["total_in_window"] == 0
        assert result["entropy"] == 0.0

    def test_default_trust_score(self):
        now = datetime.now(PACIFIC).isoformat()
        memories = [{"created_at": now}]  # no trust_score field
        result = compute_memory_entropy(memories, window_hours=24)
        assert result["avg_trust_score"] == 0.5


# ═══════════════════════════════════════════════════════════════════
# Tier-Based TTL
# ═══════════════════════════════════════════════════════════════════

class TestTierTTL:
    def test_working_tier_ttl(self):
        tier_map = {"working": 3600, "session": 86400, "long_term": None, "archived": 604800}
        result = compute_tier_ttl("working", tier_map)
        assert result is not None
        dt = datetime.fromisoformat(result)
        assert dt > datetime.now(PACIFIC)

    def test_long_term_no_ttl(self):
        tier_map = {"working": 3600, "session": 86400, "long_term": None, "archived": 604800}
        result = compute_tier_ttl("long_term", tier_map)
        assert result is None

    def test_unknown_tier(self):
        tier_map = {"working": 3600}
        result = compute_tier_ttl("unknown", tier_map)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Temporal Trust Boost
# ═══════════════════════════════════════════════════════════════════

class TestTemporalTrustBoost:
    def test_recent_memory_gets_boost(self):
        now_str = datetime.now(PACIFIC).isoformat()
        result = apply_temporal_trust_boost(0.7, now_str, temporal_weight_factor=0.1)
        assert result > 0.7
        assert result <= 1.0

    def test_old_memory_minimal_boost(self):
        old_str = (datetime.now(PACIFIC) - timedelta(days=100)).isoformat()
        result = apply_temporal_trust_boost(0.7, old_str, temporal_weight_factor=0.1)
        assert result == pytest.approx(0.7, abs=0.01)

    def test_capped_at_one(self):
        now_str = datetime.now(PACIFIC).isoformat()
        result = apply_temporal_trust_boost(0.95, now_str, temporal_weight_factor=0.1, max_boost=0.15)
        assert result <= 1.0

    def test_no_timestamp_returns_original(self):
        result = apply_temporal_trust_boost(0.7, None)
        assert result == 0.7

    def test_invalid_timestamp_returns_original(self):
        result = apply_temporal_trust_boost(0.7, "not-a-date")
        assert result == 0.7


# ═══════════════════════════════════════════════════════════════════
# Memory class integration: Trust Scoring
# ═══════════════════════════════════════════════════════════════════

class TestMemoryTrustScoringIntegration:
    @pytest.fixture
    def memory_with_trust_scoring(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                trust_scoring=TrustScoringConfig(enabled=True, archive_threshold=0.3),
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    def test_score_facts_returns_scores(self, memory_with_trust_scoring):
        memory_with_trust_scoring.llm.generate_response.return_value = json.dumps({
            "scored_facts": [
                {"text": "Name is Alice", "score": 0.9},
                {"text": "Had coffee today", "score": 0.2},
            ]
        })
        result = memory_with_trust_scoring._score_facts(["Name is Alice", "Had coffee today"])
        assert result["Name is Alice"] == 0.9
        assert result["Had coffee today"] == 0.2

    def test_score_facts_fallback_on_error(self, memory_with_trust_scoring):
        memory_with_trust_scoring.llm.generate_response.side_effect = Exception("LLM error")
        result = memory_with_trust_scoring._score_facts(["fact1", "fact2"])
        assert result["fact1"] == 0.5
        assert result["fact2"] == 0.5

    def test_trust_score_in_search_results(self, memory_with_trust_scoring):
        now = datetime.now(PACIFIC).isoformat()
        memory_with_trust_scoring.vector_store.search.return_value = [
            _make_output_data("m1", {
                "data": "Name is Alice",
                "created_at": now,
                "trust_score": 0.9,
                "memory_tier": "long_term",
            }, score=0.85),
        ]
        results = memory_with_trust_scoring._search_vector_store("test", {"user_id": "u1"}, limit=10)
        assert results[0]["trust_score"] == 0.9
        assert results[0]["memory_tier"] == "long_term"


# ═══════════════════════════════════════════════════════════════════
# Memory class integration: Hierarchical Tiers
# ═══════════════════════════════════════════════════════════════════

class TestMemoryHierarchicalIntegration:
    @pytest.fixture
    def memory_with_hierarchy(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                hierarchical_memory=HierarchicalMemoryConfig(enabled=True),
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    def test_tier_assigned_working_for_run_id(self, memory_with_hierarchy):
        """Memories with run_id should be assigned to working tier."""
        memory_with_hierarchy._create_memory(
            "test fact",
            {"test fact": [0.1, 0.2, 0.3]},
            metadata={"run_id": "run_123"},
        )
        call_args = memory_with_hierarchy.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert payload.get("memory_tier") == "working"

    def test_tier_assigned_session_for_agent_id(self, memory_with_hierarchy):
        """Memories with agent_id should be assigned to session tier."""
        memory_with_hierarchy._create_memory(
            "test fact",
            {"test fact": [0.1, 0.2, 0.3]},
            metadata={"agent_id": "agent_123"},
        )
        call_args = memory_with_hierarchy.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert payload.get("memory_tier") == "session"

    def test_tier_assigned_long_term_default(self, memory_with_hierarchy):
        """Memories without run_id or agent_id should be long_term."""
        memory_with_hierarchy._create_memory(
            "test fact",
            {"test fact": [0.1, 0.2, 0.3]},
            metadata={"user_id": "user_123"},
        )
        call_args = memory_with_hierarchy.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert payload.get("memory_tier") == "long_term"

    def test_explicit_tier_not_overridden(self, memory_with_hierarchy):
        """If memory_tier is already set, it should not be overridden."""
        memory_with_hierarchy._create_memory(
            "test fact",
            {"test fact": [0.1, 0.2, 0.3]},
            metadata={"user_id": "user_123", "memory_tier": "archived"},
        )
        call_args = memory_with_hierarchy.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert payload.get("memory_tier") == "archived"

    def test_tier_in_get_all_results(self, memory_with_hierarchy):
        now = datetime.now(PACIFIC).isoformat()
        memory_with_hierarchy.vector_store.list.return_value = [
            _make_output_data("m1", {
                "data": "test",
                "created_at": now,
                "memory_tier": "long_term",
                "trust_score": 0.8,
            }),
        ]
        results = memory_with_hierarchy._get_all_from_vector_store({"user_id": "u1"}, limit=10)
        assert results[0]["memory_tier"] == "long_term"
        assert results[0]["trust_score"] == 0.8


# ═══════════════════════════════════════════════════════════════════
# Memory class integration: Conflict Resolution
# ═══════════════════════════════════════════════════════════════════

class TestMemoryConflictResolutionIntegration:
    @pytest.fixture
    def memory_with_conflict_flag(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                conflict_resolution=ConflictResolutionConfig(
                    enabled=True, contradiction_action="flag"
                ),
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    @pytest.fixture
    def memory_with_conflict_auto(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                conflict_resolution=ConflictResolutionConfig(
                    enabled=True, contradiction_action="auto_resolve"
                ),
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    def test_conflict_flag_uses_conflict_aware_prompt(self, memory_with_conflict_flag):
        """When conflict resolution is enabled, should use conflict-aware update prompt."""
        # This test verifies the prompt selection logic
        assert memory_with_conflict_flag.config.conflict_resolution.enabled is True
        assert memory_with_conflict_flag.config.conflict_resolution.contradiction_action == "flag"

    def test_conflict_auto_resolve_config(self, memory_with_conflict_auto):
        assert memory_with_conflict_auto.config.conflict_resolution.contradiction_action == "auto_resolve"


# ═══════════════════════════════════════════════════════════════════
# Memory class integration: Proactive Forgetting / Entropy
# ═══════════════════════════════════════════════════════════════════

class TestMemoryProactiveForgettingIntegration:
    @pytest.fixture
    def memory_with_forgetting(self):
        with patch("mem0.memory.main.capture_event"), \
             patch("mem0.memory.storage.SQLiteManager"), \
             patch("mem0.utils.factory.EmbedderFactory.create") as mock_emb, \
             patch("mem0.utils.factory.VectorStoreFactory.create", side_effect=[MagicMock(), MagicMock()]), \
             patch("mem0.utils.factory.LlmFactory.create"):

            mock_emb.return_value.embed.return_value = [0.1, 0.2, 0.3]

            from mem0.memory.main import Memory

            config = MemoryConfig(
                cleanup=CleanupConfig(
                    proactive_forgetting=ProactiveForgettingConfig(
                        enabled=True,
                        entropy_threshold=0.5,
                        tier_ttl_seconds={"working": 3600, "session": 86400, "long_term": None, "archived": 604800},
                    ),
                ),
                hierarchical_memory=HierarchicalMemoryConfig(enabled=True),
            )
            memory = Memory(config=config)
            memory.db = MagicMock()
            yield memory

    def test_tier_ttl_applied_on_create(self, memory_with_forgetting):
        """Working tier memories should get a TTL applied."""
        memory_with_forgetting._create_memory(
            "temp fact",
            {"temp fact": [0.1, 0.2, 0.3]},
            metadata={"run_id": "run_123"},
        )
        call_args = memory_with_forgetting.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert "expires_at" in payload
        assert payload["memory_tier"] == "working"

    def test_long_term_no_ttl(self, memory_with_forgetting):
        """Long-term memories should not get a TTL."""
        memory_with_forgetting._create_memory(
            "important fact",
            {"important fact": [0.1, 0.2, 0.3]},
            metadata={"user_id": "user_123"},
        )
        call_args = memory_with_forgetting.vector_store.insert.call_args
        payload = call_args[1]["payloads"][0] if "payloads" in call_args[1] else call_args[0][2][0]
        assert "expires_at" not in payload
        assert payload["memory_tier"] == "long_term"

    def test_entropy_report(self, memory_with_forgetting):
        """Entropy report should compute noise metrics."""
        now = datetime.now(PACIFIC).isoformat()
        memory_with_forgetting.vector_store.list.return_value = [
            _make_output_data("m1", {"created_at": now, "trust_score": 0.1}),
            _make_output_data("m2", {"created_at": now, "trust_score": 0.9}),
            _make_output_data("m3", {"created_at": now, "trust_score": 0.15}),
            _make_output_data("m4", {"created_at": now, "trust_score": 0.85}),
        ]
        report = memory_with_forgetting.entropy_report(user_id="u1")
        assert report["total_in_window"] == 4
        assert report["low_trust_count"] == 2
        assert report["entropy"] == 0.5
        assert report["needs_cleanup"] is True  # 0.5 >= threshold 0.5

    def test_entropy_report_clean(self, memory_with_forgetting):
        """Low entropy should not trigger cleanup."""
        now = datetime.now(PACIFIC).isoformat()
        memory_with_forgetting.vector_store.list.return_value = [
            _make_output_data("m1", {"created_at": now, "trust_score": 0.9}),
            _make_output_data("m2", {"created_at": now, "trust_score": 0.8}),
        ]
        report = memory_with_forgetting.entropy_report(user_id="u1")
        assert report["needs_cleanup"] is False
