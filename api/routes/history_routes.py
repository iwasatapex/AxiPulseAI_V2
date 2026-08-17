"""Authenticated history endpoints with user isolation."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth.dependencies import current_user, require_admin
from api.database.connection import get_db
from api.database.models import DecisionHistory, PredictionHistory


router = APIRouter(dependencies=[Depends(current_user)])


@router.get("/decisions")
def decisions(
    user=Depends(current_user),
    db: Session = Depends(get_db),
):
    rows = db.query(DecisionHistory).filter(
        DecisionHistory.user == user["username"]
    ).all()
    return {
        "count": len(rows),
        "data": [
            {
                "id": row.id,
                "user": row.user,
                "engine": row.engine,
                "created_at": row.created_at,
            }
            for row in rows
        ],
    }


@router.get("/predictions")
def predictions(
    _admin=Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = db.query(PredictionHistory).all()
    return {
        "count": len(rows),
        "data": [
            {
                "id": row.id,
                "model": row.model,
                "created_at": row.created_at,
            }
            for row in rows
        ],
    }
