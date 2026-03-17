import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from mem0.embeddings.configs import EmbedderConfig
from mem0.graphs.configs import GraphStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.configs.rerankers.config import RerankerConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    trust_score: Optional[float] = Field(None, description="LLM-evaluated importance/trust score (0.0-1.0)")
    memory_tier: Optional[str] = Field(None, description="Memory tier: working, session, long_term, or archived")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class TemporalDecayConfig(BaseModel):
    """Configuration for time-based memory decay. Score = Relevance × e^(-λt)"""
    enabled: bool = Field(default=False, description="Enable temporal decay on search results")
    decay_rate: float = Field(default=0.01, description="Decay coefficient λ (per day). Higher = faster decay")
    time_field: str = Field(default="updated_at", description="Payload field to use for time. Falls back to 'created_at'")


class TTLConfig(BaseModel):
    """Configuration for event-driven TTL-based memory expiration."""
    enabled: bool = Field(default=False, description="Enable TTL-based memory expiration")
    default_ttl_seconds: Optional[int] = Field(default=None, description="Default TTL in seconds for new memories. None = no expiration")
    auto_purge_on_search: bool = Field(default=False, description="Automatically purge expired memories during search")


class CompactionConfig(BaseModel):
    """Configuration for memory summarization/compaction."""
    enabled: bool = Field(default=False, description="Enable memory compaction")
    max_memories_before_compact: int = Field(default=100, description="Trigger compaction when memories exceed this threshold")
    summary_batch_size: int = Field(default=20, description="Number of memories to summarize per batch")
    preserve_recent_hours: float = Field(default=24.0, description="Do not compact memories newer than this many hours")
    summary_prompt: Optional[str] = Field(default=None, description="Custom prompt for memory summarization LLM call")


class GarbageCollectionConfig(BaseModel):
    """Configuration for garbage collection of unused/low-weight memories."""
    enabled: bool = Field(default=False, description="Enable garbage collection")
    min_idle_days: float = Field(default=30.0, description="Minimum days since last access before a memory is GC-eligible")
    min_access_count: int = Field(default=0, description="Memories accessed fewer than this count are GC-eligible")
    score_threshold: float = Field(default=0.0, description="Memories with relevance score below this are GC-eligible (used with decay)")
    batch_size: int = Field(default=500, description="Max memories to scan per GC run")


class TrustScoringConfig(BaseModel):
    """Configuration for LLM-based trust/importance scoring of memories."""
    enabled: bool = Field(default=False, description="Enable trust scoring on new memories")
    archive_threshold: float = Field(
        default=0.3,
        description="Memories scored below this threshold are automatically archived instead of stored as active",
    )
    scoring_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for trust scoring. Must instruct LLM to return scores 0.0-1.0",
    )


class HierarchicalMemoryConfig(BaseModel):
    """Configuration for three-tier hierarchical memory management."""
    enabled: bool = Field(default=False, description="Enable hierarchical memory tiers (working/session/long_term)")
    working_memory_max_items: int = Field(
        default=10, description="Maximum number of items in working memory before oldest are promoted"
    )
    session_ttl_seconds: int = Field(
        default=86400, description="Default TTL in seconds for session-tier memories (24h)"
    )
    promotion_threshold: float = Field(
        default=0.6,
        description="Trust score above which session memories are promoted to long-term on summarization",
    )
    demotion_idle_days: float = Field(
        default=30.0, description="Idle days before long-term memories are demoted to archived"
    )


class ConflictResolutionConfig(BaseModel):
    """Configuration for memory conflict detection and resolution."""
    enabled: bool = Field(default=False, description="Enable conflict detection when updating memories")
    temporal_weight_factor: float = Field(
        default=0.1,
        description="Bonus added to trust score per unit of recency (newer memories get higher weight)",
    )
    contradiction_action: str = Field(
        default="flag",
        description="Action on contradiction: 'flag' (keep both, mark conflict) or 'auto_resolve' (keep newer, demote older)",
    )


class ProactiveForgettingConfig(BaseModel):
    """Configuration for entropy-based proactive forgetting and tier-based TTL."""
    enabled: bool = Field(default=False, description="Enable proactive forgetting mechanisms")
    entropy_threshold: float = Field(
        default=0.8,
        description="Entropy score (0-1) above which a cleanup alert is triggered. Higher = more noise tolerated",
    )
    entropy_window_hours: int = Field(
        default=24, description="Time window in hours over which entropy is computed"
    )
    tier_ttl_seconds: Dict[str, Optional[int]] = Field(
        default_factory=lambda: {"working": 3600, "session": 86400, "long_term": None, "archived": 604800},
        description="TTL in seconds per memory tier. None means no expiration",
    )


class CleanupConfig(BaseModel):
    """Unified configuration for all memory cleanup mechanisms."""
    temporal_decay: TemporalDecayConfig = Field(default_factory=TemporalDecayConfig, description="Temporal decay settings")
    ttl: TTLConfig = Field(default_factory=TTLConfig, description="TTL-based expiration settings")
    compaction: CompactionConfig = Field(default_factory=CompactionConfig, description="Memory compaction settings")
    garbage_collection: GarbageCollectionConfig = Field(default_factory=GarbageCollectionConfig, description="Garbage collection settings")
    proactive_forgetting: ProactiveForgettingConfig = Field(default_factory=ProactiveForgettingConfig, description="Proactive forgetting settings")


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem0_dir, "history.db"),
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    cleanup: CleanupConfig = Field(
        description="Configuration for memory cleanup mechanisms (decay, TTL, compaction, GC)",
        default_factory=CleanupConfig,
    )
    trust_scoring: TrustScoringConfig = Field(
        description="Configuration for LLM-based trust/importance scoring",
        default_factory=TrustScoringConfig,
    )
    hierarchical_memory: HierarchicalMemoryConfig = Field(
        description="Configuration for three-tier hierarchical memory management",
        default_factory=HierarchicalMemoryConfig,
    )
    conflict_resolution: ConflictResolutionConfig = Field(
        description="Configuration for memory conflict detection and resolution",
        default_factory=ConflictResolutionConfig,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
