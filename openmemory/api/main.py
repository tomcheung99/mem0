import patch_mem0  # must be first — patches mem0's OpenAI LLM before any usage
import asyncio
import datetime
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from app.config import ALLOWED_ORIGINS, ALLOW_CREDENTIALS, DEFAULT_APP_ID, USER_ID
from app.database import Base, SessionLocal, engine, get_database_state, set_database_state
from app.mcp_server import setup_mcp_server
from app.models import App, User
from app.routers import apps_router, backup_router, config_router, memories_router, stats_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from sqlalchemy import text

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(initialize_database_with_retry(app))
    yield


app = FastAPI(title="OpenMemory API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.db_initialized = False
app.state.db_init_error = None
app.state.db_init_status = "pending"
app.state.db_init_stage = "not-started"

# Check for USER_ID and create default user if needed
def create_default_user():
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            # Create default user
            user = User(
                id=uuid4(),
                user_id=USER_ID,
                name="Default User",
                created_at=datetime.datetime.now(datetime.UTC)
            )
            db.add(user)
            db.commit()
    finally:
        db.close()


def create_default_app():
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            return

        # Check if app already exists
        existing_app = db.query(App).filter(
            App.name == DEFAULT_APP_ID,
            App.owner_id == user.id
        ).first()

        if existing_app:
            return

        app = App(
            id=uuid4(),
            name=DEFAULT_APP_ID,
            owner_id=user.id,
            created_at=datetime.datetime.now(datetime.UTC),
            updated_at=datetime.datetime.now(datetime.UTC),
        )
        db.add(app)
        db.commit()
    finally:
        db.close()


def initialize_database_state():
    logging.info("Initializing database state")

    set_database_state(False, None, status="initializing", stage="db-connectivity-check")
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    logging.info("Database connectivity verified")
    set_database_state(False, None, status="initializing", stage="create-tables")
    Base.metadata.create_all(bind=engine)

    logging.info("Database tables verified")
    set_database_state(False, None, status="initializing", stage="create-default-user")
    create_default_user()

    logging.info("Default user verified")
    set_database_state(False, None, status="initializing", stage="create-default-app")
    create_default_app()

    set_database_state(True, None, status="ready", stage="complete")
    logging.info("Database state initialized")


async def initialize_database_with_retry(app: FastAPI, max_attempts: int = 10, delay_seconds: int = 5):
    for attempt in range(1, max_attempts + 1):
        try:
            app.state.db_init_status = "initializing"
            app.state.db_init_stage = f"attempt-{attempt}"
            await asyncio.to_thread(initialize_database_state)
            app.state.db_initialized = True
            app.state.db_init_error = None
            app.state.db_init_status = "ready"
            app.state.db_init_stage = "complete"
            return
        except Exception as exc:
            app.state.db_init_error = str(exc)
            app.state.db_init_status = "retrying" if attempt < max_attempts else "failed"
            app.state.db_init_stage = f"attempt-{attempt}"
            set_database_state(
                False,
                str(exc),
                status="retrying" if attempt < max_attempts else "failed",
                stage=f"attempt-{attempt}",
            )
            logging.exception("Database initialization attempt %s/%s failed", attempt, max_attempts)
            if attempt == max_attempts:
                return
            await asyncio.sleep(delay_seconds)


@app.get("/health")
async def healthcheck():
    database_initialized, database_init_error, database_status, database_stage = get_database_state()
    return {
        "status": "ok",
        "database_initialized": database_initialized,
        "database_init_error": database_init_error,
        "database_status": database_status,
        "database_stage": database_stage,
    }

# Setup MCP server
setup_mcp_server(app)

# Include routers
app.include_router(memories_router)
app.include_router(apps_router)
app.include_router(stats_router)
app.include_router(config_router)
app.include_router(backup_router)

# Add pagination support
add_pagination(app)
