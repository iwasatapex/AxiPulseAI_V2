"""Persistence service for API history records."""
from __future__ import annotations

from fastapi.encoders import jsonable_encoder

from api.database.connection import SessionLocal
from api.database.models import DecisionHistory, PredictionHistory


class PersistenceService:
    def save_decision(self, user, payload):
        db = SessionLocal()
        try:
            record = DecisionHistory(
                user=str(user),
                engine="ADIE",
                payload=jsonable_encoder(payload),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record.id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def save_prediction(self, model, input_data, output_data):
        db = SessionLocal()
        try:
            record = PredictionHistory(
                model=str(model),
                input_data=jsonable_encoder(input_data),
                output_data=jsonable_encoder(output_data),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record.id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def save_decision(user, payload):
    return PersistenceService().save_decision(user, payload)


def save_prediction(model, input_data, output_data):
    return PersistenceService().save_prediction(model, input_data, output_data)
