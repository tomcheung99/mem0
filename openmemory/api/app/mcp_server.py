"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import contextvars
import datetime
import json
import logging
import uuid

from app.config import MCP_API_KEY, MCP_RATE_LIMIT
from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.routing import APIRouter
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.responses import JSONResponse as _StarletteJSONResponse
from starlette.routing import Route as _StarletteRoute
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Initialize MCP
mcp = FastMCP("mem0-mcp-server")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
session_contexts: dict[uuid.UUID, tuple[str, str]] = {}

security = HTTPBearer(auto_error=False)


def get_rate_limit_key(request: Request) -> str:
    uid = request.path_params.get("user_id") or user_id_var.get(None)
    if uid:
        return uid

    if request.client and request.client.host:
        return request.client.host

    return "anonymous"


limiter = Limiter(key_func=get_rate_limit_key)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> None:
    if not MCP_API_KEY:
        return

    provided_key = None
    if credentials and credentials.scheme.lower() == "bearer":
        provided_key = credentials.credentials

    if not provided_key:
        provided_key = request.headers.get("x-api-key") or request.query_params.get("api_key")

    if provided_key != MCP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Initialize SSE transport
sse = SseServerTransport("/mcp/messages/")

# Streamable HTTP session manager (proxy-friendly alternative to SSE)
_streamable_manager: StreamableHTTPSessionManager | None = None


def get_streamable_manager() -> StreamableHTTPSessionManager:
    """Get or lazily create the StreamableHTTP session manager."""
    global _streamable_manager
    if _streamable_manager is None:
        _streamable_manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server,
            json_response=False,
            stateless=False,
        )
    return _streamable_manager


class _StreamableHTTPApp:
    """ASGI app for the Streamable HTTP transport endpoint."""

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive, send)

        # API key check (mirrors verify_api_key logic)
        if MCP_API_KEY:
            provided_key = None
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                provided_key = auth_header[7:]
            if not provided_key:
                provided_key = (
                    request.headers.get("x-api-key")
                    or request.query_params.get("api_key")
                )
            if provided_key != MCP_API_KEY:
                resp = _StarletteJSONResponse(
                    {"detail": "Invalid API Key"}, status_code=401
                )
                await resp(scope, receive, send)
                return

        # Extract path params and set context variables
        path_params = scope.get("path_params", {})
        uid = path_params.get("user_id", "")
        cname = path_params.get("client_name", "")
        user_id_var.set(uid)
        client_name_var.set(cname)

        manager = get_streamable_manager()
        await manager.handle_request(scope, receive, send)


_streamable_http_app = _StreamableHTTPApp()

@mcp.tool(description="Add a new memory. This method is called everytime the user informs anything about themselves, their preferences, or anything that has any relevant information which can be useful in the future conversation. This can also be called when the user asks you to remember something.")
async def add_memories(text: str, metadata: dict | None = None) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    logging.info("MCP add_memories called user_id=%s client_name=%s", uid, client_name)

    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        logging.warning("add_memories unavailable because memory client is not initialized")
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Check if app is active
            if not app.is_active:
                return f"Error: App {app.name} is currently paused on OpenMemory. Cannot create new memories."

            # Merge user metadata with source tracking metadata
            merged_metadata = {
                "source_app": "openmemory",
                "mcp_client": client_name,
                **(metadata or {}),
            }

            response = memory_client.add(text,
                                         user_id=uid,
                                         metadata=merged_metadata)

            # Process the response and update database
            if isinstance(response, dict) and 'results' in response:
                for result in response['results']:
                    memory_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()

                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                metadata_=metadata or {},
                                state=MemoryState.active
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']

                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if memory else None,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'UPDATE':
                        if memory:
                            memory.content = result['memory']
                            if metadata:
                                memory.metadata_ = {**(memory.metadata_ or {}), **metadata}

                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.UTC)
                            # Create history entry
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted
                            )
                            db.add(history)

                db.commit()

            return json.dumps(response)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return f"Error adding to memory: {e}"


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    logging.info("MCP search_memory called user_id=%s client_name=%s", uid, client_name)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        logging.warning("search_memory unavailable because memory client is not initialized")
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get accessible memory IDs based on ACL
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            filters = {
                "user_id": uid
            }

            embeddings = memory_client.embedding_model.embed(query, "search")

            hits = memory_client.vector_store.search(
                query=query, 
                vectors=embeddings, 
                limit=100, 
                filters=filters,
            )

            allowed = set(str(mid) for mid in accessible_memory_ids) if accessible_memory_ids else None

            results = []
            for h in hits:
                # All vector db search functions return OutputData class
                id, score, payload = h.id, h.score, h.payload
                if allowed and h.id is None or h.id not in allowed: 
                    continue
                
                results.append({
                    "id": id, 
                    "memory": payload.get("data"), 
                    "hash": payload.get("hash"),
                    "created_at": payload.get("created_at"), 
                    "updated_at": payload.get("updated_at"), 
                    "score": score,
                })

            # Keep top 10 after ACL filtering
            results = results[:10]

            for r in results: 
                if r.get("id"): 
                    access_log = MemoryAccessLog(
                        memory_id=uuid.UUID(r["id"]),
                        app_id=app.id,
                        access_type="search",
                        metadata_={
                            "query": query,
                            "score": r.get("score"),
                            "hash": r.get("hash"),
                        },
                    )
                    db.add(access_log)
            db.commit()

            return json.dumps({"results": results}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    logging.info("MCP list_memories called user_id=%s client_name=%s", uid, client_name)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        logging.warning("list_memories unavailable because memory client is not initialized")
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get all memories
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            # Filter memories based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={
                                    "hash": memory_data.get('hash')
                                }
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={
                                "hash": memory.get('hash')
                            }
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="Delete specific memories by their IDs")
async def delete_memories(memory_ids: list[str]) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    logging.info("MCP delete_memories called user_id=%s client_name=%s count=%s", uid, client_name, len(memory_ids))
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        logging.warning("delete_memories unavailable because memory client is not initialized")
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Convert string IDs to UUIDs and filter accessible ones
            requested_ids = [uuid.UUID(mid) for mid in memory_ids]
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # Only delete memories that are both requested and accessible
            ids_to_delete = [mid for mid in requested_ids if mid in accessible_memory_ids]

            if not ids_to_delete:
                return "Error: No accessible memories found with provided IDs"

            # Delete from vector store
            for memory_id in ids_to_delete:
                try:
                    memory_client.delete(str(memory_id))
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in ids_to_delete:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                if memory:
                    # Update memory state
                    memory.state = MemoryState.deleted
                    memory.deleted_at = now

                    # Create history entry
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.active,
                        new_state=MemoryState.deleted
                    )
                    db.add(history)

                    # Create access log entry
                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="delete",
                        metadata_={"operation": "delete_by_id"}
                    )
                    db.add(access_log)

            db.commit()
            return f"Successfully deleted {len(ids_to_delete)} memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp.tool()
def update_memory(memory_id: str, new_content: str) -> str:
    """Update an existing memory's content. Re-embeds in vector store and syncs SQL.

    Args:
        memory_id: The UUID of the memory to update.
        new_content: The new content/text for this memory.
    """
    try:
        # Validate UUID format
        try:
            parsed_id = uuid.UUID(memory_id)
        except ValueError:
            return f"Invalid memory_id format: {memory_id}"

        user_id = user_id_var.get("")
        client_name = client_name_var.get("")
        if not user_id:
            return "Error: user_id not found in context"

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id, client_name)

            # Find memory and check ownership
            memory = db.query(Memory).filter(
                Memory.id == parsed_id,
                Memory.state == MemoryState.active
            ).first()
            if not memory:
                return f"Memory {memory_id} not found or already deleted"

            # ACL check
            has_access = check_memory_access_permissions(db, memory, app)
            if not has_access:
                return f"App '{client_name}' does not have permission to update memory {memory_id}"

            old_content = memory.content

            # Sync to vector store (re-embed + history)
            memory_client = get_memory_client_safe()
            if memory_client:
                memory_client.update(str(parsed_id), new_content)

            # Update SQL
            memory.content = new_content
            memory.updated_at = datetime.datetime.now(datetime.UTC)
            db.commit()

            # Access log
            access_log = MemoryAccessLog(
                memory_id=parsed_id,
                app_id=app.id,
                access_type="update",
                metadata_={"old_content": old_content, "new_content": new_content}
            )
            db.add(access_log)
            db.commit()

            return f"Memory {memory_id} updated successfully"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error updating memory: {e}")
        return f"Error updating memory: {e}"


async def _handle_sse_impl(request: Request):
    """Core SSE connection handler shared across route variants."""
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    session_id: uuid.UUID | None = None

    # Wrap the ASGI send to inject anti-buffering headers into the SSE response.
    # Reverse proxies (Traefik, nginx, Cloudflare) buffer SSE by default which
    # prevents the MCP initialization handshake from completing in time.
    original_send = request._send

    async def unbuffered_send(message):
        if message.get("type") == "http.response.start":
            headers = dict(message.get("headers", []))
            extra_headers = [
                (b"x-accel-buffering", b"no"),       # nginx
                (b"cache-control", b"no-cache, no-transform"),  # general / CDN
                (b"x-content-type-options", b"nosniff"),
            ]
            message = {
                **message,
                "headers": list(message.get("headers", [])) + extra_headers,
            }
        await original_send(message)

    try:
        existing_session_ids = set(sse._read_stream_writers.keys())
        async with sse.connect_sse(
            request.scope,
            request.receive,
            unbuffered_send,
        ) as (read_stream, write_stream):
            created_session_ids = set(sse._read_stream_writers.keys()) - existing_session_ids
            if len(created_session_ids) == 1:
                session_id = created_session_ids.pop()
                session_contexts[session_id] = (uid or "", client_name or "")
                logging.info(
                    "Registered MCP session context session_id=%s user_id=%s client_name=%s",
                    session_id.hex,
                    uid,
                    client_name,
                )
            elif created_session_ids:
                logging.warning(
                    "Unexpected multiple MCP sessions created for user_id=%s client_name=%s count=%s",
                    uid,
                    client_name,
                    len(created_session_ids),
                )

            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        if session_id is not None:
            session_contexts.pop(session_id, None)
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)


@mcp_router.get("/{client_name}/sse/{user_id}")
@limiter.limit(MCP_RATE_LIMIT)
async def handle_sse(request: Request, _: None = Depends(verify_api_key)):
    """Handle SSE connections for a specific user and client"""
    await _handle_sse_impl(request)


@mcp_router.get("/{client_name}/sse/{user_id}/")
@limiter.limit(MCP_RATE_LIMIT)
async def handle_sse_trailing(request: Request, _: None = Depends(verify_api_key)):
    """Handle SSE connections (trailing slash variant)"""
    await _handle_sse_impl(request)


@mcp_router.head("/{client_name}/sse/{user_id}")
async def handle_sse_head(request: Request, _: None = Depends(verify_api_key)):
    """HEAD probe for SSE endpoint"""
    return Response(status_code=200)


@mcp_router.head("/{client_name}/sse/{user_id}/")
async def handle_sse_head_trailing(request: Request, _: None = Depends(verify_api_key)):
    """HEAD probe for SSE endpoint (trailing slash variant)"""
    return Response(status_code=200)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request, _: None = Depends(verify_api_key)):
    return await process_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_client_message(request: Request, _: None = Depends(verify_api_key)):
    return await process_post_message(request)


async def process_post_message(request: Request):
    """Handle POST messages for SSE"""
    user_token = None
    client_token = None
    try:
        uid = request.path_params.get("user_id")
        client_name = request.path_params.get("client_name")

        if not uid or not client_name:
            session_id_param = request.query_params.get("session_id")
            if session_id_param:
                try:
                    session_id = uuid.UUID(hex=session_id_param)
                except ValueError:
                    session_id = None
                if session_id is not None:
                    session_context = session_contexts.get(session_id)
                    if session_context:
                        uid, client_name = session_context
                        logging.info(
                            "Resolved MCP session context from session_id=%s user_id=%s client_name=%s",
                            session_id.hex,
                            uid,
                            client_name,
                        )

        if uid is not None:
            user_token = user_id_var.set(uid)
        if client_name is not None:
            client_token = client_name_var.set(client_name)

        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        if user_token is not None:
            user_id_var.reset(user_token)
        if client_token is not None:
            client_name_var.reset(client_token)

def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Include MCP router in the FastAPI app (SSE endpoints)
    app.include_router(mcp_router)

    # Mount Streamable HTTP transport (proxy-friendly, no persistent connection)
    for _path in (
        "/mcp/{client_name}/http/{user_id}",
        "/mcp/{client_name}/http/{user_id}/",
    ):
        app.routes.insert(
            0,
            _StarletteRoute(
                _path,
                endpoint=_streamable_http_app,
                methods=["GET", "POST", "DELETE"],
            ),
        )
