"""SQLAlchemy persistence models."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String

from api.database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionHistory(Base):
    __tablename__ = "decision_history"

    id = Column(Integer, primary_key=True)
    user = Column(String, nullable=False, index=True)
    engine = Column(String, nullable=False)
    payload = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True)
    model = Column(String, nullable=False)
    input_data = Column(JSON)
    output_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
