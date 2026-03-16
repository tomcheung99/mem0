# OpenMemory Architecture Analysis

## Executive Summary

The OpenMemory system is a FastAPI-based memory management service that uses MCP (Model Context Protocol) servers to allow external apps (like nanobot, vane) to create and search memories. The key architectural components are:

1. **API Routes** - REST endpoints for memory operations
2. **MCP Server** - Protocol endpoint for apps to connect via HTTP/SSE
3. **Database Models** - User, App, Memory, and ACL tracking
4. **Memory Client** - Integration with Mem0 for vector storage and semantic memory

---

## 1. How OpenMemory API Handles Memory Creation/Updates

### API Route: `POST /api/v1/memories/`

**Location:** `/Users/tomcheung/Project-2026/mem0/openmemory/api/app/routers/memories.py` (lines 222-347)

**Request Model:**
```python
class CreateMemoryRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    messages: Optional[List[dict]] = None
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"  # APP IDENTIFIER
```

**Key Steps in Memory Creation:**

1. **User Validation** - Looks up user by `user_id` string
2. **App Lookup/Creation** - Gets or creates app by matching:
   - `App.name == request.app` (e.g., "nanobot", "vane", "openmemory")
   - `App.owner_id == user.id`
3. **App Active Check** (🔴 **CRITICAL GATE**):
   ```python
   if not app_obj.is_active:
       raise HTTPException(status_code=403, 
           detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")
   ```
4. **Metadata Enrichment** - Adds source tracking:
   ```python
   merged_metadata = {
       "source_app": "openmemory",
       "mcp_client": request.app,
       **request.metadata,
   }
   ```
5. **Memory Client Call** - Sends to Mem0:
   ```python
   qdrant_response = memory_client.add(
       mem0_input,
       user_id=request.user_id,
       metadata=merged_metadata,
       infer=request.infer
   )
   ```
6. **Database Sync** - Processes response for ADD/UPDATE/DELETE events:
   - Creates/updates Memory records in database
   - Records MemoryStatusHistory entries
   - Links memory to the app (via `app_id` foreign key)

---

## 2. How Apps Connect to OpenMemory (MCP Server)

### MCP Server Connection Methods

OpenMemory exposes **three MCP transport mechanisms**:

#### A. **Streamable HTTP Transport** (RECOMMENDED for most apps)
- **URL Pattern:** `/mcp/{client_name}/http/{user_id}`
- **Example:** `http://localhost:8765/mcp/nanobot/http/Tom`
- **How it works:**
  1. Client establishes SSE connection to get endpoint URL
  2. Client performs request/response via HTTP POST to endpoint
  3. Preserves auth headers on POST calls
  4. **Best for:** Apps with HTTP clients, proxied environments

**Location:** `/Users/tomcheung/Project-2026/mem0/openmemory/api/app/mcp_server.py` (lines 164-177)

#### B. **SSE Transport** (Server-Sent Events)
- **URL Pattern:** `/mcp/messages/`
- **How it works:** WebSocket-like streaming for real-time message delivery
- **Best for:** Long-lived connections, less firewall-friendly

#### C. **Custom Handler** (In app code)
- Apps can call REST API directly via `/api/v1/memories/` endpoint
- No MCP required, just HTTP POST

### MCP Tool Registration

**Location:** `/Users/tomcheung/Project-2026/mem0/openmemory/api/app/mcp_server.py` (lines 179-273)

Two main MCP tools exposed:

```python
@mcp.tool()
async def add_memories(text: str, metadata: dict | None = None) -> str:
    # 1. Gets user_id and client_name from context variables
    # 2. Calls get_user_and_app(db, user_id, client_name)
    # 3. Checks app.is_active
    # 4. Calls memory_client.add()
    # 5. Processes results and updates database
    
@mcp.tool()
async def search_memory(query: str) -> str:
    # 1. Searches vector store
    # 2. Applies ACL filtering
    # 3. Logs access
    # 4. Returns top 10 results
```

**Critical Context Setup:**
```python
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
```

These context variables are **extracted from the URL path** by the middleware:
```python
class StreamableHTTPMiddleware:
    async def __call__(self, scope, receive, send):
        m = _STREAMABLE_HTTP_PATTERN.match(scope.get("path", ""))
        if m:
            # Extract from path like /mcp/nanobot/http/Tom
            path_params = m.groupdict()  # {"client_name": "nanobot", "user_id": "Tom"}
            user_id_var.set(path_params["user_id"])
            client_name_var.set(path_params["client_name"])
```

---

## 3. Configuration Requirements for Apps to Successfully Create Memories

### A. At the OpenMemory Level (API Configuration)

**Environment Variables** (in `/Users/tomcheung/Project-2026/mem0/openmemory/api/.env`):
```env
# Default user (required)
USER=<user-id>

# LLM credentials
OPENAI_API_KEY=sk-xxx
# OR
AI_GATEWAY_API_KEY=xxx
AI_GATEWAY_BASE_URL=xxx

# Mem0 vector store configuration
PG_HOST=localhost
PG_PORT=5432
PG_DB=openmemory_vectors
PG_USER=postgres
PG_PASSWORD=xxx
PG_SSLMODE=disable

# MCP Security
MCP_API_KEY=<optional, enables Bearer token auth>
MCP_RATE_LIMIT=100/minute

# CORS
ALLOWED_ORIGINS=http://localhost:3000
```

### B. App Database Prerequisites

Each app must be registered in the `apps` table:

```python
class App(Base):
    __tablename__ = "apps"
    id = Column(UUID, primary_key=True)
    owner_id = Column(UUID, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    # UNIQUE constraint: (owner_id, name)
```

**Registration Logic** (`get_or_create_app` in `/Users/tomcheung/Project-2026/mem0/openmemory/api/app/utils/db.py`):

```python
def get_or_create_app(db: Session, user: User, app_id: str) -> App:
    """Get or create an app for the given user"""
    app = db.query(App).filter(
        App.owner_id == user.id, 
        App.name == app_id
    ).first()
    if not app:
        app = App(owner_id=user.id, name=app_id)
        db.add(app)
        db.commit()
        db.refresh(app)
    return app
```

**⚠️ CRITICAL:** If `app.is_active == False`, memory creation fails with 403 error.

### C. App Configuration (Client-Side)

Apps need to know:
1. **User ID**: The string identifier for the user (default: "default_user")
2. **App Name**: The app identifier (e.g., "nanobot", "vane")
3. **OpenMemory URL**: Base URL of the API
4. **API Key** (optional): If `MCP_API_KEY` is set in OpenMemory

**Example: Nanobot Configuration** (`/Users/tomcheung/Project-2026/nanobot/config.railway.json`):

```json
{
  "tools": {
    "mcpServers": {
      "openmemory": {
        "type": "streamableHttp",
        "url": "${OPENMEMORY_MCP_URL:-http://127.0.0.1:8765/mcp/nanobot/http/Tom}",
        "headers": {
          "Authorization": "Bearer ${OPENMEMORY_MCP_API_KEY}"
        },
        "toolTimeout": 60
      }
    }
  }
}
```

**Key Components:**
- **`mcp_servers.openmemory.url`** = `/mcp/{client_name}/http/{user_id}`
  - `client_name` = "nanobot" (the app identifier in OpenMemory)
  - `user_id` = "Tom" (the user identifier)
- **`headers.Authorization`** = Bearer token matching `MCP_API_KEY` (if required)

---

## 4. App Identification in MCP Server Code

### Flow: How OpenMemory Identifies Which App is Calling

**Step 1: URL Extraction (Middleware)**
```python
# URL: /mcp/nanobot/http/Tom
# Regex pattern: ^/mcp/(?P<client_name>[^/]+)/http/(?P<user_id>[^/]+)/?$

path_params = {
    "client_name": "nanobot",  # ← This is the APP IDENTIFIER
    "user_id": "Tom"           # ← This is the USER IDENTIFIER
}
```

**Location:** `/Users/tomcheung/Project-2026/mem0/openmemory/api/app/mcp_server.py` (lines 41-42, 150-155)

**Step 2: Context Variable Assignment**
```python
user_id_var.set("Tom")      # Stored in contextvars
client_name_var.set("nanobot")  # Stored in contextvars
```

**Step 3: Tool Execution (add_memories)**
```python
@mcp.tool()
async def add_memories(text: str, metadata: dict | None = None) -> str:
    uid = user_id_var.get(None)           # "Tom"
    client_name = client_name_var.get(None)  # "nanobot"
    
    # Get or create user and app
    user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
    # This creates/retrieves: User(user_id="Tom") and App(name="nanobot", owner_id=user.id)
```

**Step 4: Database Linking**
```python
memory = Memory(
    id=memory_id,
    user_id=user.id,        # Foreign key to users.id
    app_id=app.id,          # Foreign key to apps.id  ← APP TRACKING
    content=result['memory'],
    metadata_=metadata or {},
    state=MemoryState.active
)
db.add(memory)
```

### Result
Every memory created via MCP has:
- `memory.app_id` pointing to the app that created it
- `memory.metadata_` containing source tracking:
  ```python
  {
      "source_app": "openmemory",
      "mcp_client": "nanobot"  # The app identifier
  }
  ```

---

## Database Schema (Key Tables)

### users
```
id (UUID, PK)
user_id (String, UNIQUE) - e.g., "Tom"
name (String, nullable)
created_at (DateTime)
```

### apps
```
id (UUID, PK)
owner_id (UUID, FK→users.id)
name (String) - e.g., "nanobot", "vane", "openmemory"
is_active (Boolean, default=True) ← CRITICAL
created_at (DateTime)
updated_at (DateTime)
UNIQUE(owner_id, name)
```

### memories
```
id (UUID, PK)
user_id (UUID, FK→users.id)
app_id (UUID, FK→apps.id) ← TRACKS WHICH APP CREATED IT
content (String)
metadata_ (JSON) - includes "mcp_client" value
state (Enum: active|paused|archived|deleted)
created_at (DateTime)
updated_at (DateTime)
```

### memory_access_logs
```
id (UUID, PK)
memory_id (UUID, FK→memories.id)
app_id (UUID, FK→apps.id) ← TRACKS WHICH APP ACCESSED IT
access_type (String) - "search", "view", etc.
accessed_at (DateTime)
```

---

## Why "vane" Might Not Be Creating Memories

### Potential Issues:

1. **App Not Registered or Not Active**
   - The `vane` app record in the database has `is_active = False`
   - **Fix:** Update the app status: `UPDATE apps SET is_active = TRUE WHERE name = 'vane';`

2. **Incorrect MCP URL Configuration**
   - Vane's config doesn't point to `/mcp/vane/http/{user_id}`
   - **Fix:** Update vane's config to use correct URL pattern

3. **Missing API Key**
   - If `MCP_API_KEY` is set in OpenMemory, vane's requests don't include it
   - **Fix:** Add `Authorization: Bearer <API_KEY>` header in vane's config

4. **Network Connectivity**
   - Vane can't reach OpenMemory server
   - **Fix:** Check network, URL, firewall rules

5. **Memory Client Unavailable**
   - Mem0/Qdrant/Ollama backend is down
   - **Fix:** Check `/health` endpoint, restart dependencies

6. **User Not Found**
   - Vane is sending a `user_id` that doesn't exist in database
   - **Fix:** Ensure user is created or matches expected value

### Debugging Steps:

1. Check app status:
   ```bash
   # Query the database
   sqlite3 /Users/tomcheung/Project-2026/mem0/openmemory/api/openmemory.db
   > SELECT id, name, is_active FROM apps;
   ```

2. Check memory creation logs:
   ```bash
   # Look at application logs for "Creating memory for user_id"
   ```

3. Verify connectivity:
   ```bash
   curl -H "Authorization: Bearer <API_KEY>" \
        http://localhost:8765/mcp/vane/http/Tom
   ```

4. Check health:
   ```bash
   curl http://localhost:8765/health
   ```

---

## API Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/memories/` | Create memory (REST) |
| GET | `/api/v1/memories/` | List memories |
| PUT | `/api/v1/memories/{memory_id}` | Update memory |
| POST | `/api/v1/memories/actions/archive` | Archive memory |
| POST | `/api/v1/memories/search` | Search memories |
| GET | `/api/v1/apps/` | List apps |
| GET | `/api/v1/apps/{app_id}` | Get app details |
| GET | `/api/v1/apps/{app_id}/memories` | List memories by app |
| GET | `/mcp/{client}/{transport}/{user_id}` | MCP connection (internal) |
| GET | `/health` | Health check |

---

## MCP Tool Availability

When a client connects via `/mcp/{client_name}/http/{user_id}`, two tools are available:

1. **`add_memories`** - Create or update a memory
   - Parameters: `text` (str), `metadata` (dict, optional)
   - Called by: LLM agents to persist information
   
2. **`search_memory`** - Query memories
   - Parameters: `query` (str)
   - Called by: LLM agents to recall context
   - Returns: Top 10 semantically similar memories

Both tools operate in the context of the authenticated user/app pair.

---

## Summary Table: Memory Creation Flow

| Stage | Component | Key Logic |
|-------|-----------|-----------|
| **1. HTTP Request** | REST API or MCP Client | POST to `/api/v1/memories/` or call `add_memories()` |
| **2. User Resolution** | `get_or_create_user()` | Find/create user by `user_id` |
| **3. App Resolution** | `get_or_create_app()` | Find/create app by `(owner_id, name)` |
| **4. Permission Check** | `app.is_active` | 🔴 **GATE:** Must be `True` |
| **5. Metadata Enrichment** | Memory route/MCP tool | Add source tracking: `{"source_app": "openmemory", "mcp_client": "..."}` |
| **6. Vector Storage** | Memory Client (Mem0) | Call `memory_client.add()` → Qdrant/vector store |
| **7. DB Sync** | Memory route/MCP tool | Create Memory record with `app_id` foreign key |
| **8. Response** | MCP Tool or REST API | Return created memory details |

