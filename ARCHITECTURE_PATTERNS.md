# Mem0 Architecture Patterns & Design Decisions

## Key Architectural Patterns

### 1. **Factory Pattern** (Pluggable Providers)
**File**: `utils/factory.py`

All major components use factory pattern for extensibility:
```python
LlmFactory.create("openai", config)           # 23+ LLM providers
EmbedderFactory.create("openai", config)      # 18 embedder providers  
VectorStoreFactory.create("qdrant", config)   # 27 vector store providers
RerankerFactory.create("cohere", config)      # 6 reranker providers
GraphStoreFactory.create("neo4j", config)     # 3 graph DB providers
```

**Benefit**: Easy to swap providers without changing core logic

---

### 2. **Abstract Base Classes** (Polymorphism)
- `MemoryBase` → `Memory` & `AsyncMemory` (memory/base.py)
- `VectorStoreBase` → Qdrant, Chroma, Pinecone, etc. (vector_stores/base.py)
- `LlmBase` → OpenAI, Anthropic, Ollama, etc. (llms/)
- `EmbedderBase` → OpenAI, HuggingFace, etc. (embeddings/)
- `BaseReranker` → Cohere, LLM, HuggingFace, etc. (reranker/)

**Benefit**: Consistent interface, easy extension

---

### 3. **Concurrent Execution** (Thread Pool)
**File**: `main.py`, Lines 376-383, 700-708, 839-847

```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(self._add_to_vector_store, ...)
    future2 = executor.submit(self._add_to_graph, ...)
    concurrent.futures.wait([future1, future2])
    
    result1 = future1.result()
    result2 = future2.result()
```

**Usage Points**:
- Add: Vector store + Graph store in parallel
- Search: Vector search + Graph search in parallel
- Get all: Vector retrieval + Graph retrieval in parallel

**Benefit**: Performance optimization for I/O-bound operations

---

### 4. **Pydantic Models** (Validation & Serialization)
**File**: `configs/base.py`

```python
class MemoryItem(BaseModel):
    id: str
    memory: str
    hash: Optional[str]
    metadata: Optional[Dict[str, Any]]
    score: Optional[float]
    created_at: Optional[str]
    updated_at: Optional[str]

class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig
    llm: LlmConfig
    embedder: EmbedderConfig
    cleanup: CleanupConfig
```

**Benefit**: Type safety, automatic validation, JSON serialization

---

### 5. **Strategy Pattern** (Cleanup Mechanisms)
**File**: `cleanup.py`

Four independent cleanup strategies:
1. **Temporal Decay** - Score degradation over time
2. **TTL** - Hard expiration timestamps
3. **Compaction** - Memory summarization
4. **Garbage Collection** - Removal of unused memories

Each strategy:
- Is optional (can be enabled/disabled independently)
- Has its own configuration
- Can be mixed and matched

**Benefit**: Flexible, composable memory lifecycle management

---

### 6. **Dual-Mode Inference** (LLM-Powered vs Raw)
**File**: `main.py`, Lines 393-428 (infer=False) vs 430-604 (infer=True)

**Mode 1: Raw (infer=False)**
```
Message → Embed → Store directly
```

**Mode 2: LLM-Powered (infer=True)**
```
Message → Extract Facts (LLM) → Search existing → 
Determine Actions (LLM) → ADD/UPDATE/DELETE/NONE
```

**Benefit**: Flexibility - use fast raw mode or intelligent LLM mode

---

### 7. **Session Scoping** (Multi-Tenant Ready)
**File**: `main.py`, Lines 94-172 (_build_filters_and_metadata)

Three orthogonal session identifiers:
- **user_id** - End user
- **agent_id** - AI agent
- **run_id** - Execution run

Requirements:
- At least one required
- Can combine (e.g., user_id + agent_id + run_id)
- All scoped to same filters for isolation

**Benefit**: Multi-tenant capable, flexible scoping granularity

---

### 8. **History Tracking** (Audit Trail)
**File**: `storage.py`, Lines 10-190

SQLite table records every mutation:
```sql
CREATE TABLE history (
    id, memory_id, old_memory, new_memory, 
    event, created_at, updated_at, actor_id, role
)
```

On every operation:
- `_create_memory()` → INSERT history with event="ADD"
- `_update_memory()` → INSERT history with event="UPDATE"
- `_delete_memory()` → INSERT history with event="DELETE"

**Benefit**: Compliance, debugging, reproducibility

---

### 9. **Metadata Layering** (Flexible Custom Fields)
**File**: `main.py`, Lines 632-657 (get), 744-759 (get_all), 989-1004 (search)

Core fields (promoted):
```python
promoted_keys = ["user_id", "agent_id", "run_id", "actor_id", "role"]
```

Internal fields (hidden):
```python
internal_keys = {"data", "hash", "created_at", "updated_at", "id"}
```

Everything else → custom metadata dict:
```python
additional_metadata = {k: v for k, v in payload.items() 
                      if k not in core_and_promoted_keys}
```

**Benefit**: Fixed schema with unlimited custom fields

---

### 10. **Embedding Caching** (Efficiency)
**File**: `main.py`, Lines 479-481, 1114-1117, 1220-1223

Existing embeddings reused when available:
```python
if data in existing_embeddings:
    embeddings = existing_embeddings[data]
else:
    embeddings = self.embedding_model.embed(data, "add")
```

Used in:
- Fact extraction (search for existing memories)
- Memory action determination
- Update operations

**Benefit**: Reduces API calls to embedding service

---

### 11. **Dual Memory Extraction** (Context-Aware)
**File**: `main.py`, Lines 267-286, configs/prompts.py

Automatic selection based on message roles:
```python
if agent_id present AND assistant role in messages:
    → Use AGENT_MEMORY_EXTRACTION_PROMPT
      (extract assistant's characteristics/capabilities)
else:
    → Use USER_MEMORY_EXTRACTION_PROMPT
      (extract user preferences/facts)
```

Two prompts in `configs/prompts.py`:
- Lines 62-120: USER_MEMORY_EXTRACTION_PROMPT (user messages only)
- Lines 123-180: AGENT_MEMORY_EXTRACTION_PROMPT (assistant messages only)

**Benefit**: Context-appropriate fact extraction

---

### 12. **Filter Processing** (Advanced Operators)
**File**: `main.py`, Lines 865-959

```python
def _has_advanced_operators(filters) → bool
def _process_metadata_filters(filters) → dict
```

Conversion pipeline:
```
User input: {"role": "user", "score": {"gte": 0.7}}
        ↓
Process: Extract operators (gte) → Validate
        ↓
Output: Vector store compatible format
```

Supports:
- Comparison: eq, ne, gt, gte, lt, lte
- Membership: in, nin
- String: contains, icontains
- Logical: AND, OR, NOT
- Wildcard: "*"

**Benefit**: Rich filtering without vector store knowledge

---

### 13. **Search Pipeline Stages** (Composable Processing)
**File**: `main.py`, Lines 961-1027

Stages applied in sequence:
1. Vector embedding (Lines 962)
2. Vector search (Line 963)
3. TTL filtering (Lines 984-987)
4. Result formatting (Lines 989-1004)
5. Temporal decay (Lines 1010-1012)
6. Access tracking (Lines 1023-1025)
7. Reranking (Lines 853-858 - search method level)

Each stage can be independently enabled/disabled in config.

**Benefit**: Customizable search behavior

---

### 14. **Async/Sync Duality** (Both Patterns)
**File**: `main.py` (Memory sync), `main.py` (AsyncMemory async)

Both classes implement same interface:
- Sync: `Memory.add(), .get(), .search(), .update(), .delete()`
- Async: `AsyncMemory.add(), .get(), .search(), .update(), .delete()`

Internal `_*` methods work for both:
- Sync methods call sync versions
- Async methods call async versions with await

**Benefit**: Flexibility for sync and async applications

---

### 15. **LLM Function Calling** (Structured Output)
**File**: `main.py`, Lines 441-447 (fact extraction), 508-526 (action determination)

Uses JSON mode for structured output:
```python
response = self.llm.generate_response(
    messages=[...],
    response_format={"type": "json_object"}  # Forces JSON
)
```

Then parses:
```python
parsed = json.loads(response)
facts = parsed["facts"]  # or parsed["memory"]
```

**Benefit**: Reliable extraction of structured data from LLM

---

## Design Decisions

### Vector Store Abstraction
**Decision**: Support 27+ vector stores via abstract interface  
**Rationale**: 
- No single "best" vector store (price, latency, features vary)
- Users have existing infrastructure
- Allows comparison and migration
**Cost**: Lowest common denominator interface (filters, metadata handling)

### Temporal Decay at Search Time
**Decision**: Apply decay during search, not during storage  
**Rationale**:
- Scores remain immutable in vector store
- Multiple decay rates possible
- Can enable/disable dynamically
**Cost**: Slight computation overhead per search

### TTL as Explicit Timestamps
**Decision**: expires_at field in payload, check at search time  
**Rationale**:
- Vector store agnostic (no server-side TTL required)
- Can be set differently per memory
- Matches common TTL pattern
**Cost**: Requires cleanup during search

### SQLite for History
**Decision**: Use SQLite for audit trail  
**Rationale**:
- No external dependencies
- File-based, portable
- Sufficient for history lookups
**Cost**: Limited to single-machine (can't shard)

### LLM Function Calling for Updates
**Decision**: Use LLM to determine ADD/UPDATE/DELETE/NONE actions  
**Rationale**:
- Better than fixed rules (e.g., similarity threshold)
- Catches semantic relationships human rules miss
- Flexible for custom logic (custom prompts)
**Cost**: Two LLM calls per add (fact extraction + action determination)

### Metadata Layering
**Decision**: Promote session IDs, hide internals, rest as custom metadata  
**Rationale**:
- Clean public API (MemoryItem)
- Flexibility for extensions
- Backward compatibility
**Cost**: Complexity in formatting/parsing

### Graph Store as Optional
**Decision**: Graph entirely optional, parallel to vector store  
**Rationale**:
- Graph not needed for all use cases
- Adds complexity, not always value
- Concurrent with vector operations (no performance hit when disabled)
**Cost**: Duplicate filtering/scoping logic

### Session Scoping
**Decision**: Require at least one of user_id, agent_id, run_id  
**Rationale**:
- Multi-tenant isolation required
- Prevent accidental data leaks
- Flexible scoping levels
**Cost**: More verbose API (can't get all memories without filter)

---

## Performance Considerations

1. **Concurrent Operations**
   - Vector + Graph operations in parallel
   - Reduces add/search latency

2. **Embedding Caching**
   - Avoid re-embedding same text
   - Speeds up fact extraction searches

3. **Search Pipeline Stages**
   - TTL filter before formatting (skip expired early)
   - Reranking after vector search (best of both)

4. **Batch Cleanup**
   - Compaction batches 20 memories per LLM call
   - GC scans in batches of 500

5. **Session Scoping**
   - Filters reduce vector search scope
   - Prevents full-DB scans

---

## Extensibility Points

1. **Custom LLM Prompts**
   - `custom_fact_extraction_prompt` in MemoryConfig
   - `custom_update_memory_prompt` in MemoryConfig
   - `summary_prompt` in CompactionConfig

2. **Custom Providers**
   - Inherit from base classes
   - Register in factory (or use custom factory)

3. **Custom Cleanup Strategies**
   - Implement cleanup logic in hooks (future)
   - Currently: temporal_decay + TTL + compaction + GC

4. **Custom Metadata**
   - Any key-value pair in metadata dict
   - Survives updates (except where explicitly overwritten)

5. **Custom Reranking**
   - Implement BaseReranker
   - Swap in via config

---

## Security Considerations

1. **Metadata Filtering**
   - Session IDs scoped at storage level
   - Can't query cross-user without explicit user_id
   
2. **Telemetry**
   - Sensitive tokens redacted (auth, credentials, passwords, keys)
   - User IDs hashed with MD5 in events

3. **History Records**
   - Immutable audit trail
   - Tracks actor_id and role

4. **Graph Store**
   - Optional (can disable for privacy)
   - Adds entity tracking overhead

---

## Trade-offs Summary

| Feature | Benefit | Cost |
|---------|---------|------|
| 27+ Vector Stores | Flexibility | Interface lowest common denominator |
| Temporal Decay | Prevent old data dominating | Computation per search |
| TTL Timestamps | Flexible per-memory expiration | Requires search-time checking |
| LLM Function Calling | Better updates than rules | Extra LLM call, latency |
| Async/Sync Both | Works everywhere | Code duplication |
| Session Scoping | Multi-tenant safe | API verbosity |
| Optional Graph | Rich relationships | Duplicate logic |
| Embedding Caching | Fewer API calls | Memory overhead |

