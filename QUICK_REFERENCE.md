# Mem0 Architecture - Quick Reference

## Key Findings Summary

### 1. DIRECTORY STRUCTURE
- **Core**: `mem0/memory/main.py` (2600 lines) - Memory class orchestrator
- **Config**: `mem0/configs/base.py` - MemoryConfig with 4 cleanup strategies
- **Providers**: 
  - 27 vector stores (Qdrant default)
  - 23+ LLMs (OpenAI default)
  - 18 embedders (OpenAI default)
  - 6 rerankers (optional)
  - 3 graph DBs (Neo4j/Memgraph/Kuzu)

### 2. MEMORY LIFECYCLE

**ADD** (Lines 288-604):
```
Input messages → Session validation → 
Procedural? Yes → LLM summary → Create memory
Procedural? No → Extract facts (LLM) → Search existing → 
Determine actions (LLM: ADD/UPDATE/DELETE/NONE) → Execute
```
- **infer=False**: Raw message storage
- **infer=True**: LLM-powered fact extraction & update
- Dual path extraction:
  - Agent memory (assistant messages): AGENT_MEMORY_EXTRACTION_PROMPT
  - User memory (user messages): USER_MEMORY_EXTRACTION_PROMPT

**RETRIEVE** (Lines 617-763):
```
get(id) → Fetch from vector store → Format MemoryItem
get_all() → List with filters and limit → Format results
```

**SEARCH** (Lines 765-1027):
```
Embed query → Vector search with filters → 
TTL cleanup → Format results → Temporal decay →
Access tracking (GC) → Reranking (optional) → Return
```

**UPDATE** (Lines 1029-1049):
```
Embed new data → Get existing metadata → 
Update data+hash+timestamp → History record
```

**DELETE** (Lines 1051-1097):
```
Single: Get existing → Delete → History record
All: List by filters → Delete each → Reset store
```

### 3. DATA SCHEMA

**MemoryItem** (What users see):
```python
{
  "id": str,              # UUID
  "memory": str,          # Text content
  "hash": str,            # MD5(data)
  "created_at": str,      # ISO 8601 (US/Pacific)
  "updated_at": str,      # ISO 8601 (US/Pacific)
  "score": float,         # 0.0-1.0 relevance
  "user_id": str,         # Session ID
  "agent_id": str,        # Session ID (optional)
  "run_id": str,          # Session ID (optional)
  "actor_id": str,        # Creator (optional)
  "role": str,            # "user"|"assistant"
  "metadata": {}          # Custom fields
}
```

**Vector Store Payload** (Internal):
```python
{
  "data": str,
  "hash": str,
  "created_at": str,
  "updated_at": str,
  "user_id|agent_id|run_id": str,  # Session scoping
  "actor_id": str,
  "role": str,
  "memory_type": str,             # semantic|episodic|procedural
  "expires_at": str,              # TTL field
  "access_count": int,            # GC field
  "last_accessed_at": str,        # GC field
  "[custom]": any                 # User metadata
}
```

### 4. VECTOR STORE

**Base Interface** (`vector_stores/base.py`):
- `insert(vectors, payloads, ids)`
- `search(query, vectors, limit, filters)` → List[SearchResult]
- `get(vector_id)`, `update()`, `delete()`
- `list(filters, limit)`, `reset()`

**27 Providers**: Qdrant (default), Chroma, PGVector, Pinecone, MongoDB, Milvus, Baidu, Cassandra, Neptune, Upstash, Azure (AI Search, MySQL), Databricks, Elasticsearch, OpenSearch, Supabase, Weaviate, FAISS, Redis/Valkey, S3 Vectors, LangChain

### 5. LLM USAGE

**Fact Extraction** (Lines 441-447):
- Prompt: USER_MEMORY_EXTRACTION_PROMPT (user messages only)
- or AGENT_MEMORY_EXTRACTION_PROMPT (assistant messages)
- Input: Parsed conversation
- Output: JSON {"facts": ["fact1", "fact2"]}
- Response format: JSON mode

**Memory Action Determination** (Lines 508-526):
- Prompt: DEFAULT_UPDATE_MEMORY_PROMPT
- Input: Existing memories + new facts
- Output: JSON {"memory": [{"id": "...", "text": "...", "event": "ADD|UPDATE|DELETE|NONE"}]}

**Procedural Memory** (Lines 1151-1188):
- Prompt: PROCEDURAL_MEMORY_SYSTEM_PROMPT (complete execution history)
- Input: Conversation
- Output: Structured summary with steps and results

### 6. CONFIGURATION

**MemoryConfig** (`configs/base.py:70-110`):
```python
vector_store: VectorStoreConfig
llm: LlmConfig
embedder: EmbedderConfig
graph_store: GraphStoreConfig (optional)
reranker: RerankerConfig (optional)
cleanup: CleanupConfig
  ├─ temporal_decay: TemporalDecayConfig
  ├─ ttl: TTLConfig
  ├─ compaction: CompactionConfig
  └─ garbage_collection: GarbageCollectionConfig
```

**Initialization**:
```python
from mem0 import Memory
memory = Memory()  # Uses defaults
memory = Memory.from_config({...})  # Custom config
```

### 7. SCORING & RANKING

**Vector Score**:
- Cosine/Euclidean similarity (0.0-1.0)
- From vector store search

**Temporal Decay** (cleanup.py:54-86):
- Formula: `score *= e^(-λ × days_old)`
- Default λ = 0.01/day
- After 100 days: score = 37% of original
- Applied at search time only
- Re-sorts results

**Reranking** (Lines 853-858):
- 6 rerankers: Cohere, HuggingFace, LLM, ONNX, SentenceTransformer, ZeroEntropy
- Applied after vector search if enabled
- Produces rerank_score

**Filtering**:
- Session IDs: user_id, agent_id, run_id (required)
- Advanced operators: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains
- Logical: AND, OR, NOT
- Wildcard: `{"key": "*"}`

### 8. MEMORY CLASSES

**Hierarchy**:
```
MemoryBase (abstract)
├── Memory (sync)
└── AsyncMemory (async)
```

**MemoryBase Interface** (base.py:4-64):
```python
class MemoryBase:
    def get(memory_id)
    def get_all()
    def update(memory_id, data)
    def delete(memory_id)
    def history(memory_id)
```

**Memory Attributes**:
```python
config: MemoryConfig
embedding_model: EmbedderBase
vector_store: VectorStoreBase
llm: LlmBase
reranker: Optional[BaseReranker]
graph: Optional[MemoryGraph]
db: SQLiteManager  # History
```

### 9. SEARCH PIPELINE

```
Query String
  ↓ [embed]
Vector
  ↓ [vector_store.search(filters)]
SearchResult{id, score, payload}
  ↓ [TTL check + format]
MemoryItem
  ↓ [temporal_decay if enabled]
MemoryItem{score adjusted}
  ↓ [access_tracking if GC enabled]
MemoryItem{last_accessed_at++}
  ↓ [rerank if enabled]
MemoryItem{rerank_score}
  ↓
Return {"results": [...]}
```

### 10. MEMORY TYPES

**Enum** (`configs/enums.py:4-8`):
```python
class MemoryType(Enum):
    SEMANTIC = "semantic_memory"
    EPISODIC = "episodic_memory"
    PROCEDURAL = "procedural_memory"
```

**Detection** (main.py:267-286):
- Automatic based on message roles
- User messages → User memory extraction
- Agent+assistant messages → Agent memory extraction
- Procedural requires explicit `memory_type="procedural_memory"` + agent_id

**Procedural Memory** (main.py:1151-1188):
- Full execution history with steps and results
- Used for agent workflow recording
- Non-conversational

**Graph Layers** (optional, graph_memory.py):
- Entities: Named entities from text
- Relations: Semantic relationships
- Context: Contextual linking

### 11. HISTORY & AUDIT

**SQLite Manager** (storage.py:10-190):
```python
add_history(memory_id, old_memory, new_memory, event, created_at, updated_at, actor_id, role)
get_history(memory_id) → List[HistoryRecord]
```

**Table Schema**:
```sql
CREATE TABLE history (
  id, memory_id, old_memory, new_memory, event, 
  created_at, updated_at, is_deleted, actor_id, role
)
```

**Event Types**: ADD, UPDATE, DELETE

### 12. CLEANUP MECHANISMS

**Temporal Decay** (cleanup.py:54-86):
- Search-time score adjustment
- λ = 0.01/day default
- Exponential degradation

**TTL** (cleanup.py:91-110):
- Explicit expiration: expires_at field
- Default: 30 days
- auto_purge_on_search: delete on search hit

**Compaction** (cleanup.py:115-133):
- Memory summarization via LLM
- Trigger: 100+ memories default
- Preserve: 24h+ recent memories
- Batch: 20 at a time

**Garbage Collection** (cleanup.py:138-150):
- Remove unused memories
- Eligibility: idle ≥30 days AND accessed <0 times
- Tracks: access_count, last_accessed_at
- GC-eligible marked but needs periodic purge

---

## Critical File Map

| Task | File | Lines |
|------|------|-------|
| Add memory | main.py | 288-604 |
| Get memory | main.py | 617-658 |
| Search | main.py | 765-1027 |
| Update | main.py | 1029-1242 |
| Delete | main.py | 1051-1097 |
| Config | configs/base.py | 70-110 |
| Schema | configs/base.py | 17-27 |
| Enums | configs/enums.py | 4-8 |
| Prompts | configs/prompts.py | 1-500 |
| Decay | cleanup.py | 54-86 |
| TTL | cleanup.py | 91-110 |
| Compaction | cleanup.py | 115-133 |
| GC | cleanup.py | 138-150 |
| History | storage.py | 10-190 |
| Graph | graph_memory.py | 76-152 |
| Vector Base | vector_stores/base.py | 1-59 |
| Reranker Base | reranker/base.py | 1-20 |
| Factory | utils/factory.py | 29-233 |

