"""Database engine and session configuration."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./axipulse.db")

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
