import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_READY = False
DATABASE_INIT_ERROR = None
DATABASE_INIT_STATUS = "pending"
DATABASE_INIT_STAGE = "not-started"

# load .env file (make sure you have DATABASE_URL set)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./openmemory.db")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment")

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
elif DATABASE_URL.startswith("postgresql"):
    engine_kwargs["connect_args"] = {
        "connect_timeout": 5,
        "options": "-c lock_timeout=5000 -c statement_timeout=10000",
    }
    engine_kwargs["pool_pre_ping"] = True

# SQLAlchemy engine & session
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def set_database_state(
    is_ready: bool,
    error: str | None = None,
    status: str | None = None,
    stage: str | None = None,
):
    global DATABASE_READY, DATABASE_INIT_ERROR, DATABASE_INIT_STATUS, DATABASE_INIT_STAGE
    DATABASE_READY = is_ready
    DATABASE_INIT_ERROR = error
    if status is not None:
        DATABASE_INIT_STATUS = status
    if stage is not None:
        DATABASE_INIT_STAGE = stage


def get_database_state() -> tuple[bool, str | None, str, str]:
    return DATABASE_READY, DATABASE_INIT_ERROR, DATABASE_INIT_STATUS, DATABASE_INIT_STAGE

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
