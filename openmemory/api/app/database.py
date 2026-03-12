import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_READY = False
DATABASE_INIT_ERROR = None

# load .env file (make sure you have DATABASE_URL set)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./openmemory.db")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in environment")

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

# SQLAlchemy engine & session
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def set_database_state(is_ready: bool, error: str | None = None):
    global DATABASE_READY, DATABASE_INIT_ERROR
    DATABASE_READY = is_ready
    DATABASE_INIT_ERROR = error


def get_database_state() -> tuple[bool, str | None]:
    return DATABASE_READY, DATABASE_INIT_ERROR

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
