import os
import time
import psutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from db.database import SessionLocal, DB_PATH


# Router setup
router = APIRouter(prefix="/health", tags=["Health Check"])

# Track process start time for uptime
_PROCESS_START = time.time()


# Response models

class ComponentStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: str          # healthy | degraded | unhealthy
    timestamp: str
    uptime_s: int
    version: str
    database: ComponentStatus
    model: ComponentStatus
    memory: ComponentStatus


# Helpers

def _check_database() -> ComponentStatus:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()

        size_kb = (
            round(DB_PATH.stat().st_size / 1024, 1)
            if DB_PATH.exists()
            else 0
        )

        return ComponentStatus(
            ok=True,
            detail=f"SQLite reachable — {size_kb} KB"
        )

    except Exception as exc:
        return ComponentStatus(ok=False, detail=str(exc))


def _check_model() -> ComponentStatus:
    try:
        # ensure model is importable
        from ml.simulator import model, scaler

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models" / "lstm_model.keras"
        )
        scaler_path = (
            Path(__file__).resolve().parent.parent
            / "models" / "scaler.pkl"
        )

        if not model_path.exists():
            return ComponentStatus(ok=False, detail="lstm_model.keras not found")

        if not scaler_path.exists():
            return ComponentStatus(ok=False, detail="scaler.pkl not found")

        size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)

        return ComponentStatus(
            ok=True,
            detail=f"LSTM loaded — {size_mb} MB"
        )

    except Exception as exc:
        return ComponentStatus(ok=False, detail=str(exc))


def _check_memory() -> ComponentStatus:
    mem = psutil.virtual_memory()

    pct = mem.percent
    available_mb = round(mem.available / (1024 * 1024), 1)

    return ComponentStatus(
        ok=pct < 90.0,
        detail=f"{pct:.1f}% used — {available_mb} MB available"
    )


# Endpoints

@router.get(
    "/quick",
    summary="Liveness probe",
    description="Returns 200 if process is alive (for Docker / load balancers).",
)
def liveness():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get(
    "",
    response_model=HealthResponse,
    summary="Full system health check",
    description="Checks DB, model, and memory status.",
)
def full_health():
    db_status  = _check_database()
    ml_status  = _check_model()
    mem_status = _check_memory()

    failures = sum(
        1 for s in [db_status, ml_status, mem_status] if not s.ok
    )

    if failures == 0:
        overall = "healthy"
    elif failures == 1:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_s=int(time.time() - _PROCESS_START),
        version="2.0.0",
        database=db_status,
        model=ml_status,
        memory=mem_status,
    )