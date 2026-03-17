from enum import Enum


class MemoryType(Enum):
    SEMANTIC = "semantic_memory"
    EPISODIC = "episodic_memory"
    PROCEDURAL = "procedural_memory"


class MemoryTier(Enum):
    WORKING = "working"
    SESSION = "session"
    LONG_TERM = "long_term"
    ARCHIVED = "archived"
