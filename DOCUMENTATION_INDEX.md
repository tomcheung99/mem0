# Mem0 Architecture Documentation Index

This directory contains comprehensive documentation of the Mem0 Python package architecture.

## Documents

### 1. **MEM0_ARCHITECTURE_ANALYSIS.md** (975 lines, 31 KB)
**Comprehensive technical reference covering all aspects of the architecture**

Detailed sections:
- **Section 1**: Directory structure with file descriptions
- **Section 2**: Complete memory lifecycle (add, retrieve, update, delete)
  - 2.1: Adding memories with LLM-based inference
  - 2.2: Retrieving single/all memories
  - 2.3: Searching with temporal decay and reranking
  - 2.4: Updating memory content
  - 2.5: Deleting individual/batch memories
- **Section 3**: Data models and schemas (MemoryItem, Vector Store payload)
- **Section 4**: Vector store integration (base class, 27 providers)
- **Section 5**: LLM integration (fact extraction, action determination, procedural)
- **Section 6**: Configuration system (factory pattern, setup)
- **Section 7**: Scoring, ranking, and filtering mechanisms
- **Section 8**: Memory class relationships and interfaces
- **Section 9**: Search and retrieval pipeline
- **Section 10**: Memory types (semantic, episodic, procedural) and layers
- **Section 11**: History and audit trail (SQLite)
- **Section 12**: Cleanup mechanisms (decay, TTL, compaction, GC)
- **Summary Table**: File paths and line numbers

**Use when**: You need detailed line-by-line reference, exact implementation details

---

### 2. **QUICK_REFERENCE.md** (450+ lines, 7 KB)
**Quick lookup guide for common tasks and key concepts**

Sections:
- **Directory Structure**: Key modules and their purposes
- **Memory Lifecycle**: Flow diagrams for CRUD operations
- **Data Schema**: MemoryItem and internal payload structures
- **Vector Store**: Base interface and 27 providers
- **LLM Usage**: Three usage points with examples
- **Configuration**: MemoryConfig structure and initialization
- **Scoring & Ranking**: Vector scores, temporal decay, reranking filters
- **Memory Classes**: Class hierarchy and attributes
- **Search Pipeline**: Step-by-step process from query to results
- **Memory Types**: Enums, detection, procedural, graph layers
- **History & Audit**: SQLite schema and operations
- **Cleanup Mechanisms**: All four strategies (decay, TTL, compaction, GC)
- **Critical File Map**: Quick table of important files and line numbers

**Use when**: You need a quick lookup, code examples, high-level overview

---

### 3. **ARCHITECTURE_PATTERNS.md** (350+ lines, 12 KB)
**Design patterns, architectural decisions, and trade-offs**

Key Patterns:
1. Factory Pattern - Pluggable providers
2. Abstract Base Classes - Polymorphism
3. Concurrent Execution - Thread pools
4. Pydantic Models - Validation
5. Strategy Pattern - Cleanup mechanisms
6. Dual-Mode Inference - LLM-powered vs raw
7. Session Scoping - Multi-tenant ready
8. History Tracking - Audit trails
9. Metadata Layering - Fixed + custom fields
10. Embedding Caching - Efficiency
11. Dual Memory Extraction - Context-aware
12. Filter Processing - Advanced operators
13. Search Pipeline Stages - Composable
14. Async/Sync Duality - Both patterns
15. LLM Function Calling - Structured output

Design Decisions:
- Vector store abstraction (why 27?)
- Temporal decay at search time
- TTL as explicit timestamps
- SQLite for history
- LLM for action determination
- Metadata layering
- Optional graph store
- Session scoping requirements

Performance & Security considerations

**Use when**: You're designing extensions, optimizing, understanding design rationale

---

## How to Use This Documentation

### For Understanding the System
1. Start with **QUICK_REFERENCE.md** sections 1-3 for overview
2. Read **ARCHITECTURE_PATTERNS.md** for design principles
3. Use **MEM0_ARCHITECTURE_ANALYSIS.md** for detailed reference

### For Implementing Features
1. Check **QUICK_REFERENCE.md** "Critical File Map" for relevant files
2. Reference **MEM0_ARCHITECTURE_ANALYSIS.md** sections for specific operations
3. Review **ARCHITECTURE_PATTERNS.md** for extensibility points

### For Debugging
1. Use **QUICK_REFERENCE.md** "Search Pipeline" for understanding flow
2. Check **MEM0_ARCHITECTURE_ANALYSIS.md** section 11 for history/audit
3. Reference file paths in "Critical File Map" for quick location

### For Optimizing
1. Review **ARCHITECTURE_PATTERNS.md** "Performance Considerations"
2. Check **QUICK_REFERENCE.md** "Scoring & Ranking" for tuning parameters
3. Read cleanup mechanisms in **MEM0_ARCHITECTURE_ANALYSIS.md** section 12

---

## Key Findings at a Glance

### Core Statistics
- **Main File**: `memory/main.py` (2600 lines)
- **Supported Vector Stores**: 27 providers
- **Supported LLMs**: 23+ providers
- **Memory Types**: 3 (semantic, episodic, procedural)
- **Cleanup Strategies**: 4 (decay, TTL, compaction, GC)
- **Rerankers**: 6 implementations

### Critical Operations
| Operation | File | Lines |
|-----------|------|-------|
| Add memory | main.py | 288-604 |
| Search | main.py | 765-1027 |
| Update | main.py | 1029-1242 |
| Delete | main.py | 1051-1097 |

### Key Concepts
- **Session Scoping**: user_id, agent_id, run_id (at least one required)
- **Memory Extraction**: Dual prompts (user vs agent)
- **Temporal Decay**: `score *= e^(-0.01 * days_old)`
- **TTL**: Optional expiration (default 30 days)
- **Compaction**: Summarize 20 at a time via LLM
- **GC**: Remove idle ≥30 days, accessed <threshold

---

## File Cross-References

### Configuration Files
- `mem0/configs/base.py` - MemoryConfig, MemoryItem
- `mem0/configs/enums.py` - MemoryType enum
- `mem0/configs/prompts.py` - LLM prompts
- `mem0/configs/vector_stores/` - 27 vector store configs
- `mem0/configs/llms/` - LLM configs
- `mem0/configs/embeddings/` - Embedder configs
- `mem0/configs/rerankers/` - Reranker configs

### Core Memory
- `mem0/memory/main.py` - Memory & AsyncMemory classes
- `mem0/memory/base.py` - MemoryBase interface
- `mem0/memory/cleanup.py` - TTL, decay, compaction, GC
- `mem0/memory/storage.py` - SQLiteManager (history)
- `mem0/memory/utils.py` - Parsing, extraction utilities

### Integration
- `mem0/vector_stores/` - 27 implementations
- `mem0/llms/` - 23+ LLM implementations
- `mem0/embeddings/` - 18 embedding implementations
- `mem0/reranker/` - 6 reranker implementations
- `mem0/graphs/` - Neo4j/Memgraph/Kuzu graph support

### API Client
- `mem0/client/main.py` - MemoryClient & AsyncMemoryClient
- `mem0/client/project.py` - Project management

### Utilities
- `mem0/utils/factory.py` - Factory pattern implementations
- `mem0/exceptions.py` - Custom exceptions

---

## Quick Facts

### Memory Storage Schema
```python
{
  "id": str,              # UUID
  "memory": str,          # Text content
  "hash": str,            # MD5 for dedup
  "score": float,         # 0.0-1.0 relevance
  "user_id": str,         # Session ID
  "created_at": str,      # ISO 8601 (US/Pacific)
  "updated_at": str,      # ISO 8601 (US/Pacific)
  "metadata": {}          # Custom fields
}
```

### LLM Integration Points
1. **Fact Extraction**: Extract facts from messages
   - File: main.py:441-447
   - Output: `{"facts": [...]}`

2. **Action Determination**: Decide ADD/UPDATE/DELETE/NONE
   - File: main.py:508-526
   - Output: `{"memory": [{"id": "...", "event": "..."}]}`

3. **Procedural Memory**: Summarize execution
   - File: main.py:1151-1188
   - Output: Structured execution summary

### Cleanup Strategies
- **Decay**: e^(-λt) at search time (default λ=0.01/day)
- **TTL**: Explicit expires_at (default 30 days)
- **Compaction**: LLM summarization (100+ memories)
- **GC**: Remove unused (idle ≥30 days, accessed <threshold)

---

## Related Documentation

Within the repository:
- `README.md` - Project overview
- `LLM.md` - LLM provider documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `MIGRATION_GUIDE_v1.0.md` - API migration info

---

## Contact & Contributing

See `CONTRIBUTING.md` for contribution guidelines.

---

**Last Updated**: March 2025
**Coverage**: mem0 v1.1+ architecture
**Status**: Complete architecture documentation
