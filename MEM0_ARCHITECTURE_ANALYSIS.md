# Mem0 Python Package - Architecture Analysis

## 1. Overall Directory Structure

```
/Users/tomcheung/Project-2026/mem0/mem0/
├── __init__.py                 # Package exports: Memory, AsyncMemory, MemoryClient, AsyncMemoryClient
├── client/                     # API client implementation
│   ├── main.py                 # MemoryClient and AsyncMemoryClient classes
│   ├── project.py              # Project management
│   └── utils.py                # Client utilities
├── memory/                     # Core memory management
│   ├── main.py                 # Memory & AsyncMemory classes (main orchestrator)
│   ├── base.py                 # MemoryBase abstract class
│   ├── storage.py              # SQLiteManager for history tracking
│   ├── cleanup.py              # TTL, decay, compaction, GC mechanisms
│   ├── graph_memory.py         # Graph-based memory (Neo4j integration)
│   ├── kuzu_memory.py          # Kuzu graph database variant
│   ├── memgraph_memory.py      # Memgraph variant
│   ├── setup.py                # Configuration initialization
│   ├── telemetry.py            # Event tracking
│   └── utils.py                # Utility functions for parsing, extraction
├── configs/                    # Configuration management
│   ├── base.py                 # MemoryConfig, MemoryItem, CleanupConfig
│   ├── enums.py                # MemoryType enum (SEMANTIC, EPISODIC, PROCEDURAL)
│   ├── prompts.py              # LLM prompts for fact extraction & updates
│   ├── embeddings/             # Embedding provider configs
│   ├── llms/                   # LLM provider configs
│   ├── rerankers/              # Reranker provider configs
│   └── vector_stores/          # Vector store provider configs
├── embeddings/                 # Embedding implementations
│   ├── base.py
│   ├── openai.py
│   ├── azure_openai.py
│   ├── huggingface.py
│   └── [16 embedding providers]
├── llms/                       # LLM implementations
│   ├── base.py
│   ├── openai.py
│   └── [23 LLM providers]
├── reranker/                   # Reranking implementations
│   ├── base.py                 # BaseReranker abstract class
│   ├── cohere_reranker.py
│   ├── huggingface_reranker.py
│   ├── llm_reranker.py
│   ├── onnx_reranker.py
│   ├── sentence_transformer_reranker.py
│   └── zero_entropy_reranker.py
├── vector_stores/              # Vector store implementations
│   ├── base.py                 # VectorStoreBase abstract class
│   ├── configs.py              # VectorStoreConfig
│   ├── qdrant.py               # Qdrant (default)
│   ├── chroma.py
│   ├── pgvector.py
│   ├── pinecone.py
│   ├── mongodb.py
│   └── [27 vector store providers]
├── graphs/                     # Graph store implementations
│   ├── configs.py              # GraphStoreConfig
│   ├── tools.py                # Graph extraction tools
│   ├── utils.py                # Graph utilities
│   └── neptune/
├── utils/                      # Factory & utility functions
│   └── factory.py              # Factories for LLM, Embedder, VectorStore, GraphStore, Reranker
└── exceptions.py               # Custom exception classes
```

---

## 2. Memory Lifecycle: Add, Retrieve, Update, Delete

### 2.1 ADDING MEMORIES

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 288-604

```python
Memory.add(
    messages,
    user_id=None,
    agent_id=None,
    run_id=None,
    metadata=None,
    infer=True,
    memory_type=None,
    prompt=None
)
```

**Flow:**
1. **Input Validation** (Lines 338-365):
   - Requires at least one of: user_id, agent_id, run_id
   - Normalizes messages to list format: `[{"role": "...", "content": "..."}]`
   - Validates memory_type (only "procedural_memory" is supported as explicit type)

2. **Metadata & Filtering** (Lines 338-343):
   - `_build_filters_and_metadata()` creates:
     - `processed_metadata`: Template with session IDs for storage
     - `effective_filters`: Scoped filters for vector search

3. **Branching by Memory Type** (Lines 367-391):
   - **Procedural Memory** (if agent_id + memory_type="procedural_memory"):
     - Calls `_create_procedural_memory()` (Line 1151-1188)
   - **Regular Memories** (default path):
     - Concurrent execution:
       - `_add_to_vector_store()` (Lines 393-604)
       - `_add_to_graph()` (Lines 606-615)

4. **Add to Vector Store** (Lines 393-604):

   **If `infer=False`** (Raw memory mode, Lines 394-428):
   - Iterate each message
   - Generate embedding for each message content
   - Create memory with `_create_memory()` returning memory ID
   - Return list of created memories with event="ADD"

   **If `infer=True`** (LLM-based inference, Lines 430-604):
   - **Step A: Extract Facts** (Lines 430-463):
     - Determine memory extraction mode based on `_should_use_agent_memory_extraction()`:
       - TRUE if: agent_id present AND messages contain assistant role → Use AGENT_MEMORY_EXTRACTION_PROMPT
       - FALSE otherwise → Use USER_MEMORY_EXTRACTION_PROMPT
     - Prompt LLM with system prompt + parsed messages
     - Extract JSON response → list of `new_retrieved_facts`
   
   - **Step B: Search for Existing Memories** (Lines 468-495):
     - For each new fact, embed it
     - Search vector store for similar memories (limit=5) within session scope
     - Deduplicate retrieved memories by ID
   
   - **Step C: Determine Memory Actions** (Lines 503-528):
     - Call LLM with function-calling prompt containing:
       - Existing memories (deduplicated)
       - New facts
     - LLM returns JSON with actions: ADD, UPDATE, DELETE, or NONE
   
   - **Step D: Execute Actions** (Lines 531-604):
     - **ADD**: Create new memory with `_create_memory()`
     - **UPDATE**: Update existing memory with `_update_memory()`
     - **DELETE**: Delete memory with `_delete_memory()`
     - **NONE**: If metadata has agent_id or run_id, update session IDs only

5. **Add to Graph** (Lines 606-615):
   - If graph enabled, extract entities and relationships from messages
   - Store in Neo4j/Memgraph/Kuzu graph database

**Return Format:**
```json
{
  "results": [
    {"id": "uuid", "memory": "fact text", "event": "ADD|UPDATE|DELETE"},
    ...
  ],
  "relations": [...] // if graph enabled
}
```

---

### 2.2 RETRIEVING MEMORIES

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 617-763

#### 2.2.1 Get Single Memory by ID

```python
Memory.get(memory_id: str) -> dict
```

**Flow** (Lines 617-658):
1. Fetch from vector store using `vector_store.get(vector_id=memory_id)`
2. Format result:
   - Extract "data" field → "memory" key
   - Extract promoted keys: user_id, agent_id, run_id, actor_id, role
   - Extract remaining metadata into "metadata" dict
3. Return MemoryItem (Pydantic model at line 17-27 of base.py)

**Data Schema (MemoryItem):**
```python
class MemoryItem(BaseModel):
    id: str                          # UUID
    memory: str                      # The actual text content
    hash: Optional[str]              # MD5 hash of content
    metadata: Optional[Dict[str, Any]]  # Custom metadata
    score: Optional[float]           # Relevance score from search
    created_at: Optional[str]        # ISO 8601 timestamp
    updated_at: Optional[str]        # ISO 8601 timestamp
```

#### 2.2.2 Get All Memories

```python
Memory.get_all(
    user_id=None,
    agent_id=None,
    run_id=None,
    filters=None,
    limit=100
) -> dict
```

**Flow** (Lines 660-763):
1. Validate at least one of user_id, agent_id, run_id provided
2. Concurrent execution:
   - `_get_all_from_vector_store()` with filters and limit
   - `graph.get_all()` if graph enabled
3. Format memories same as `get()`, return list under "results" key

---

### 2.3 SEARCHING MEMORIES

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 765-1027

```python
Memory.search(
    query: str,
    user_id=None,
    agent_id=None,
    run_id=None,
    limit=100,
    filters=None,
    threshold=None,
    rerank=True
) -> dict
```

**Flow:**
1. **Build Filters** (Lines 809-823):
   - Session scoping via session IDs
   - Advanced metadata filters with operators:
     - Exact: `{"key": "value"}`
     - Comparison: `{"key": {"eq"|"ne"|"gt"|"gte"|"lt"|"lte": value}}`
     - List: `{"key": {"in"|"nin": [values]}}`
     - String: `{"key": {"contains"|"icontains": text}}`
     - Logical: `{"AND": [...]}, {"OR": [...]}, {"NOT": [...]}`

2. **Vector Search** (Lines 839-863):
   - Embed query using embedding_model
   - Search vector store with filters and limit
   - Concurrent graph search if enabled

3. **TTL Cleanup** (Lines 984-1021):
   - If TTL enabled, skip expired memories (expires_at field)
   - If auto_purge_on_search, delete expired memories
   - Track access for GC (access_count, last_accessed_at)

4. **Temporal Decay** (Lines 1010-1012):
   - If decay enabled: `adjusted_score = original_score × e^(-λ × days_old)`
   - Re-sort by adjusted score

5. **Reranking** (Lines 852-858):
   - If reranker configured and rerank=True, rerank results
   - Reranker.rerank(query, memories, limit) returns reranked list

6. **Threshold Filtering** (Line 1006):
   - Filter by minimum score threshold if provided

**Threshold & Score Fields:**
- Vector score = cosine/euclidean similarity (0.0-1.0 typically)
- After temporal decay, score may decrease
- Rerank score may be separate field depending on reranker

---

### 2.4 UPDATING MEMORIES

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 1029-1242

```python
Memory.update(memory_id: str, data: str) -> dict
```

**Flow** (Lines 1029-1049):
1. Embed new data
2. Call `_update_memory()` (Lines 1190-1242):
   - Get existing memory to preserve metadata
   - Update "data" and "hash" fields
   - Preserve original created_at, update updated_at
   - Preserve session IDs (user_id, agent_id, run_id, actor_id, role)
   - Generate new embeddings
   - Update in vector store
   - Record in history database (old value → new value, event="UPDATE")
3. Return success message

---

### 2.5 DELETING MEMORIES

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 1051-1097

```python
Memory.delete(memory_id: str) -> dict
Memory.delete_all(user_id=None, agent_id=None, run_id=None) -> dict
```

**Flow:**
1. **Single Delete** (Lines 1051-1060):
   - Call `_delete_memory()` (Lines 1244-1254)
   - Get existing memory, extract old content
   - Delete from vector store
   - Record in history (event="DELETE")

2. **Delete All** (Lines 1062-1097):
   - Require at least one session ID filter
   - List all memories matching filters
   - Delete each memory individually
   - Reset vector store collection
   - Reset graph store if enabled

---

## 3. Data Model & Schema

### 3.1 Memory Item Schema

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/configs/base.py`, Lines 17-27

```python
class MemoryItem(BaseModel):
    id: str                              # UUID unique identifier
    memory: str                          # The actual text content
    hash: Optional[str]                  # MD5(data) for deduplication
    metadata: Optional[Dict[str, Any]]   # Arbitrary key-value pairs
    score: Optional[float]               # Relevance score (0.0-1.0)
    created_at: Optional[str]            # ISO 8601 (US/Pacific timezone)
    updated_at: Optional[str]            # ISO 8601 (US/Pacific timezone)
```

### 3.2 Vector Store Payload Schema

**Internal storage (what's stored in vector_store):**
```python
payload = {
    "data": str,                    # Memory text content
    "hash": str,                    # MD5 hash
    "created_at": str,              # ISO 8601 timestamp
    "updated_at": str,              # ISO 8601 timestamp
    "user_id": str,                 # Session scoping
    "agent_id": str,                # Session scoping
    "run_id": str,                  # Session scoping
    "actor_id": str,                # Who created/modified (optional)
    "role": str,                    # "user"|"assistant" (optional)
    "memory_type": str,             # "semantic_memory"|"episodic_memory"|"procedural_memory"
    # TTL fields (if enabled):
    "expires_at": str,              # ISO 8601, expiration timestamp
    # GC fields (if enabled):
    "access_count": int,            # Number of times accessed in search
    "last_accessed_at": str,        # ISO 8601 timestamp
    # Custom metadata:
    "[custom_key]": Any,            # User-provided metadata
}
```

### 3.3 Configuration Classes

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/configs/base.py`, Lines 70-110

```python
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig         # Vector DB config
    llm: LlmConfig                          # Language model config
    embedder: EmbedderConfig                # Embedding model config
    history_db_path: str                    # SQLite history DB path
    graph_store: GraphStoreConfig           # Optional graph DB config
    reranker: Optional[RerankerConfig]      # Optional reranker config
    cleanup: CleanupConfig                  # TTL, decay, compaction, GC settings
    version: str = "v1.1"                   # API version
    custom_fact_extraction_prompt: Optional[str]    # Override prompt
    custom_update_memory_prompt: Optional[str]      # Override prompt
```

**Cleanup Config** (Lines 30-68):
```python
class TemporalDecayConfig:
    enabled: bool                           # Enable decay
    decay_rate: float = 0.01                # λ coefficient (per day)
    time_field: str = "updated_at"          # Age calculation field

class TTLConfig:
    enabled: bool                           # Enable TTL expiration
    default_ttl_seconds: Optional[int]      # Expiration duration
    auto_purge_on_search: bool              # Delete expired during search

class CompactionConfig:
    enabled: bool                           # Enable memory summarization
    max_memories_before_compact: int = 100  # Trigger threshold
    summary_batch_size: int = 20            # Batch size for LLM
    preserve_recent_hours: float = 24.0     # Don't compact new memories

class GarbageCollectionConfig:
    enabled: bool                           # Enable GC
    min_idle_days: float = 30.0             # Idle threshold
    min_access_count: int = 0               # Access threshold
    score_threshold: float = 0.0            # Score threshold (with decay)
    batch_size: int = 500                   # Scan batch size
```

---

## 4. Vector Store Integration

### 4.1 Vector Store Base Class

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/vector_stores/base.py`

```python
class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(name, vector_size, distance): pass      # Create collection
    
    @abstractmethod
    def insert(vectors, payloads=None, ids=None): pass     # Add vectors
    
    @abstractmethod
    def search(query, vectors, limit=5, filters=None): pass # Semantic search
    
    @abstractmethod
    def delete(vector_id): pass                             # Delete one
    
    @abstractmethod
    def update(vector_id, vector=None, payload=None): pass # Update vector/metadata
    
    @abstractmethod
    def get(vector_id): pass                                # Fetch by ID
    
    @abstractmethod
    def list(filters=None, limit=None): pass               # List all
    
    @abstractmethod
    def list_cols(): pass                                   # List collections
    
    @abstractmethod
    def delete_col(): pass                                  # Delete collection
    
    @abstractmethod
    def col_info(): pass                                    # Collection info
    
    @abstractmethod
    def reset(): pass                                       # Delete and recreate
```

### 4.2 Supported Vector Stores

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/configs/vector_stores/configs.py`, Lines 13-37

- qdrant (default)
- chroma
- pgvector
- pinecone
- mongodb
- milvus
- baidu
- cassandra
- neptune
- upstash_vector
- azure_ai_search
- azure_mysql
- redis (valkey)
- databricks
- elasticsearch
- vertex_ai_vector_search
- opensearch
- supabase
- weaviate
- faiss
- langchain
- s3_vectors

### 4.3 Search Return Format

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 961-1027

Vector store search returns objects with:
```python
class SearchResult:
    id: str              # Memory ID (UUID)
    score: float         # Similarity score (0.0-1.0)
    payload: Dict       # Contains metadata and data fields
```

---

## 5. LLM Integration

### 5.1 LLM Base Class

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/llms/base.py`

```python
class LlmBase(ABC):
    @abstractmethod
    def generate_response(messages, response_format=None): pass
```

### 5.2 LLM Usage Points

1. **Fact Extraction** (Lines 441-447):
   - Prompt: USER_MEMORY_EXTRACTION_PROMPT or AGENT_MEMORY_EXTRACTION_PROMPT
   - Input: Parsed messages
   - Output: JSON with "facts" array
   - Response format: `{"type": "json_object"}`

2. **Memory Action Determination** (Lines 508-526):
   - Prompt: DEFAULT_UPDATE_MEMORY_PROMPT (via get_update_memory_messages)
   - Input: Existing memories + new facts
   - Output: JSON with memory array of {id, text, event, old_memory}
   - Events: ADD, UPDATE, DELETE, NONE

3. **Procedural Memory Creation** (Lines 1151-1188):
   - Prompt: PROCEDURAL_MEMORY_SYSTEM_PROMPT
   - Input: Conversation history
   - Output: Summarized procedural memory

### 5.3 Supported LLM Providers

23+ providers including: openai, azure_openai, gpt4all, ollama, anthropic, mistral, groq, vertex_ai, bedrock, etc.

---

## 6. Configuration System

### 6.1 Factory Pattern

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/utils/factory.py`

```python
LlmFactory.create(provider: str, config: dict) -> LlmBase
EmbedderFactory.create(provider: str, config: dict, vector_store_config: dict) -> EmbedderBase
VectorStoreFactory.create(provider: str, config: dict) -> VectorStoreBase
GraphStoreFactory.create(provider: str, config: MemoryConfig) -> GraphStore
RerankerFactory.create(provider: str, config: dict) -> BaseReranker
```

### 6.2 Configuration Loading

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/setup.py`

- Loads from environment variables or config files
- Sets up mem0_dir: `~/.mem0` or MEM0_DIR env var
- Default vector store: Qdrant (local)
- Default embedding: OpenAI
- Default LLM: OpenAI

### 6.3 Memory Initialization

```python
config = MemoryConfig(
    vector_store=VectorStoreConfig(provider="qdrant"),
    llm=LlmConfig(provider="openai"),
    embedder=EmbedderConfig(provider="openai"),
    cleanup=CleanupConfig(
        temporal_decay=TemporalDecayConfig(enabled=True, decay_rate=0.01),
        ttl=TTLConfig(enabled=True, default_ttl_seconds=2592000),
        compaction=CompactionConfig(enabled=False),
        garbage_collection=GarbageCollectionConfig(enabled=False)
    )
)
memory = Memory(config=config)
```

---

## 7. Scoring, Ranking & Filtering Mechanisms

### 7.1 Scoring

**Vector Score (from search):**
- Cosine similarity: range [0, 1], higher = more relevant
- Calculated by vector store based on embedding distance

**Temporal Decay Score:**
- Formula: `adjusted_score = original_score × e^(-λ × days_old)`
- File: `/Users/tomcheung/Project-2026/mem0/mem0/memory/cleanup.py`, Lines 54-86
- Only applied on search results if `cleanup.temporal_decay.enabled=True`
- decay_rate (λ) = 0.01 per day by default
- Memories older than 100 days: score ≈ 37% of original

**Access Tracking:**
- Garbage collection tracks access_count and last_accessed_at
- Updated during search if GC enabled
- File: `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 1023-1025

### 7.2 Reranking

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/reranker/base.py`

```python
class BaseReranker(ABC):
    def rerank(
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        # Returns documents with rerank_score added
```

**Supported Rerankers:**
1. **Cohere Reranker** - Third-party API
2. **HuggingFace Reranker** - Cross-encoder models
3. **LLM Reranker** - Custom LLM-based reranking
4. **ONNX Reranker** - Local inference
5. **Sentence Transformer Reranker** - Semantic similarity
6. **Zero Entropy Reranker** - Diversity-based filtering

**Usage** (Line 853-858):
```python
if rerank and self.reranker and original_memories:
    reranked_memories = self.reranker.rerank(query, original_memories, limit)
    original_memories = reranked_memories
```

### 7.3 Filtering

**Session-based Filtering:**
- user_id, agent_id, run_id automatically scoped
- All queries require at least one

**Advanced Metadata Filters:**
- Supported operators: eq, ne, gt, gte, lt, lte, in, nin, contains, icontains
- Logical operators: AND, OR, NOT
- Wildcard: `{"key": "*"}`
- File: Lines 865-959

**Example:**
```python
memory.search(
    query="...",
    user_id="user123",
    filters={
        "AND": [
            {"role": "user"},
            {"memory_type": "semantic_memory"},
            {"score": {"gte": 0.7}}
        ]
    }
)
```

---

## 8. Memory Class Relationships

### 8.1 Class Hierarchy

```
MemoryBase (Abstract)
    ├── Memory (Sync)
    └── AsyncMemory (Async)
```

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/base.py` (lines 1-64)

**MemoryBase Interface:**
```python
class MemoryBase(ABC):
    @abstractmethod
    def get(memory_id) -> dict
    
    @abstractmethod
    def get_all() -> dict
    
    @abstractmethod
    def update(memory_id, data) -> dict
    
    @abstractmethod
    def delete(memory_id) -> None
    
    @abstractmethod
    def history(memory_id) -> list
```

### 8.2 Memory Class

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 179-241

**Attributes:**
```python
class Memory(MemoryBase):
    config: MemoryConfig
    custom_fact_extraction_prompt: Optional[str]
    custom_update_memory_prompt: Optional[str]
    embedding_model: EmbedderBase              # Embedding provider
    vector_store: VectorStoreBase              # Vector DB
    llm: LlmBase                               # Language model
    db: SQLiteManager                          # History database
    reranker: Optional[BaseReranker]           # Optional reranker
    graph: Optional[MemoryGraph]               # Optional graph store
    enable_graph: bool
    collection_name: str                       # Vector store collection
    api_version: str = "v1.1"
```

### 8.3 AsyncMemory Class

Same interface as Memory but with async/await for all I/O operations.

---

## 9. Search & Retrieval Mechanism

### 9.1 Search Pipeline

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/main.py`, Lines 765-1027

```
Query Input
    ↓
[1] Embedding Generation (Lines 962)
    embedding_model.embed(query, "search") → vector
    ↓
[2] Vector Store Search (Line 963)
    vector_store.search(query, vectors, limit, filters)
    ← Returns: List[SearchResult{id, score, payload}]
    ↓
[3] TTL Filtering (Lines 984-987)
    If TTL enabled, skip expired memories
    ↓
[4] Format Results (Lines 989-1004)
    Extract data, hash, timestamps, session IDs, metadata
    ↓
[5] Temporal Decay (Lines 1010-1012)
    If decay enabled, adjust scores: score *= e^(-λt)
    Re-sort by adjusted score
    ↓
[6] Access Tracking (Lines 1023-1025)
    If GC enabled, increment access_count and last_accessed_at
    ↓
[7] Reranking (Lines 853-858)
    If reranker enabled, rerank(query, memories, limit)
    ↓
[8] Return Results
    {"results": [...], "relations": [...] if graph enabled}
```

### 9.2 Search Result Format

```python
{
    "id": str,                      # UUID
    "memory": str,                  # Text content
    "hash": str,                    # MD5 hash
    "created_at": str,              # ISO 8601
    "updated_at": str,              # ISO 8601
    "user_id": str,                 # Session ID
    "agent_id": str,                # Session ID (optional)
    "run_id": str,                  # Session ID (optional)
    "actor_id": str,                # Actor (optional)
    "role": str,                    # "user"|"assistant" (optional)
    "score": float,                 # Adjusted relevance (0.0-1.0)
    "metadata": {...}               # Custom fields
}
```

---

## 10. Memory Types, Categories & Layers

### 10.1 Memory Types

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/configs/enums.py`

```python
class MemoryType(Enum):
    SEMANTIC = "semantic_memory"        # Facts, preferences, knowledge
    EPISODIC = "episodic_memory"        # Events, experiences, conversations
    PROCEDURAL = "procedural_memory"    # How-tos, procedures, workflows
```

### 10.2 Memory Type Detection

**Automatic Classification:**
- Default memory type inferred from context
- **User messages** → stored as semantic/episodic
- **Assistant messages** → detected via agent_id + assistant role → agent memory

**File:** Lines 267-286 (Memory._should_use_agent_memory_extraction)

```python
def _should_use_agent_memory_extraction(messages, metadata):
    # TRUE if: agent_id present AND messages contain assistant role
    # Used to decide between:
    #   - USER_MEMORY_EXTRACTION_PROMPT (extract user facts)
    #   - AGENT_MEMORY_EXTRACTION_PROMPT (extract agent characteristics)
```

### 10.3 Procedural Memory

**File:** Lines 1151-1188

- Explicitly created with `memory_type="procedural_memory"` and agent_id
- Uses PROCEDURAL_MEMORY_SYSTEM_PROMPT (Lines 325-402 of prompts.py)
- Stores complete agent execution history with:
  - Task objective
  - Progress status
  - Sequential numbered steps with actions and results
  - Metadata: findings, navigation history, errors, context
- Not automatically created from conversations

**Example Creation:**
```python
memory.add(
    messages=[...],
    agent_id="agent_001",
    memory_type="procedural_memory"
)
```

### 10.4 Graph-based Memory Layers (Optional)

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/graph_memory.py`

When graph store enabled:
- **Entities Layer**: Named entities (persons, places, organizations)
- **Relations Layer**: Semantic relationships between entities
- **Context Layer**: Contextual information linking entities

**Graph Operations:**
- `add(data, filters)` → Extract entities and relations → Store in Neo4j
- `search(query, filters, limit)` → Find related entities
- `get_all(filters, limit)` → List all entities

---

## 11. History & Audit Trail

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/storage.py`, Lines 10-190

```python
class SQLiteManager:
    def add_history(
        memory_id: str,
        old_memory: Optional[str],
        new_memory: str,
        event: str,      # "ADD", "UPDATE", "DELETE"
        created_at: str,
        updated_at: Optional[str] = None,
        actor_id: Optional[str] = None,
        role: Optional[str] = None
    )
    
    def get_history(memory_id: str) -> List[Dict]
```

**History Table Schema (SQLite):**
```sql
CREATE TABLE history (
    id           TEXT PRIMARY KEY,
    memory_id    TEXT,
    old_memory   TEXT,
    new_memory   TEXT,
    event        TEXT,           -- "ADD", "UPDATE", "DELETE"
    created_at   DATETIME,
    updated_at   DATETIME,
    is_deleted   INTEGER,
    actor_id     TEXT,
    role         TEXT
)
```

**Usage:**
```python
# Get history of a memory
history = memory.history("memory_id_123")
# Returns: [
#   {id, memory_id, old_memory, new_memory, event, created_at, updated_at, actor_id, role},
#   ...
# ]
```

---

## 12. Cleanup Mechanisms

### 12.1 Temporal Decay

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/cleanup.py`, Lines 54-86

**Mechanism:**
- Applied at search time only
- Score degradation: `score *= e^(-λ × days_since_update)`
- Configuration:
  ```python
  decay_config = TemporalDecayConfig(
      enabled=True,
      decay_rate=0.01,        # λ = 0.01 per day
      time_field="updated_at" # or "created_at"
  )
  ```
- After 100 days: score × 0.37 (37% of original)
- Memories re-sorted by adjusted score

### 12.2 TTL (Time-to-Live)

**File:** Lines 108-110, 91-105

**Mechanism:**
- Explicit expiration timestamps in memory payload: `expires_at`
- Set at creation: `expires_at = now + ttl_seconds`
- Configuration:
  ```python
  ttl_config = TTLConfig(
      enabled=True,
      default_ttl_seconds=2592000,  # 30 days
      auto_purge_on_search=True     # Delete on search
  )
  ```
- Checked during search (Line 984-987):
  - If expired and auto_purge=True, delete immediately
  - Otherwise, skip in results

### 12.3 Memory Compaction

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/cleanup.py`, Lines 115-133

**Mechanism:**
- Summarize many fine-grained memories into dense blocks
- Triggered when memory count exceeds threshold
- Configuration:
  ```python
  compaction_config = CompactionConfig(
      enabled=True,
      max_memories_before_compact=100,   # Trigger at 100 memories
      summary_batch_size=20,              # Summarize 20 at a time
      preserve_recent_hours=24.0          # Don't compact < 24h old
  )
  ```
- Prompt: `build_compaction_prompt(memories)` calls LLM to create summaries
- Output format: `{"summaries": ["summary1", "summary2", ...]}`

### 12.4 Garbage Collection

**File:** `/Users/tomcheung/Project-2026/mem0/mem0/memory/cleanup.py`, Lines 138-150, and main.py Lines 1023-1025

**Mechanism:**
- Remove unused/low-scoring memories
- Configuration:
  ```python
  gc_config = GarbageCollectionConfig(
      enabled=True,
      min_idle_days=30.0,        # Unused for 30+ days
      min_access_count=0,        # Accessed < 0 times
      score_threshold=0.0,       # Score < 0.0 (with decay)
      batch_size=500             # Scan 500 at a time
  )
  ```
- Eligibility (all conditions must be true):
  - Not accessed for ≥ min_idle_days
  - Access count < min_access_count
  - (Optionally) decayed score < threshold
- Fields tracked:
  - `access_count`: Incremented on each search hit
  - `last_accessed_at`: ISO 8601 timestamp
  - Both initialized at memory creation

---

## Summary Table

| Component | File Path | Purpose |
|-----------|-----------|---------|
| **Memory Class** | `main.py:179-2600` | Core memory orchestrator |
| **Base Interface** | `base.py:1-64` | Abstract MemoryBase |
| **Config** | `configs/base.py:70-110` | MemoryConfig with all settings |
| **Data Schema** | `configs/base.py:17-27` | MemoryItem Pydantic model |
| **Vector Store** | `vector_stores/base.py` | 27+ implementations |
| **LLM Integration** | `memory/main.py:430-526` | Fact extraction & actions |
| **Reranking** | `reranker/base.py` | 6 reranker implementations |
| **Temporal Decay** | `cleanup.py:54-86` | Score degradation over time |
| **TTL System** | `cleanup.py:91-110` | Expiration timestamps |
| **Compaction** | `cleanup.py:115-133` | Memory summarization |
| **Garbage Collection** | `cleanup.py:138-150` | Unused memory removal |
| **Graph Memory** | `graph_memory.py:1-100` | Entity/relation tracking |
| **History** | `storage.py:10-190` | SQLite audit trail |
| **Prompts** | `configs/prompts.py:1-500` | LLM instructions |
| **Factory** | `utils/factory.py` | Provider instantiation |

