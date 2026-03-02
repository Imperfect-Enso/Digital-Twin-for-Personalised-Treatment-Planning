from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import json

from core.scoring import score_and_rank
from ml.simulator import TREATMENT_EFFECTS
from db.database import get_db, SimulationResult

router = APIRouter(prefix="/compare", tags=["Treatment Comparison"])


# -----------------------------------------------------------------------------
# Request Schema
# -----------------------------------------------------------------------------

class VitalReading(BaseModel):
    blood_pressure: float = Field(..., example=91.4)
    glucose:        float = Field(..., example=108.3)
    heart_rate:     float = Field(..., example=77.2)
    inflammation:   float = Field(..., example=1.80)


class CompareRequest(BaseModel):
    patient_id:     int | None = Field(None, example=1)
    vitals_history: list[VitalReading] = Field(..., min_length=3)


def parse_vitals(vitals: list[VitalReading]) -> list[dict]:
    return [v.model_dump() for v in vitals]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/treatments",
    summary="List all available treatments",
    description="Returns all treatment options and their monthly effects on vitals."
)
def list_treatments():
    return {
        "total":      len(TREATMENT_EFFECTS),
        "treatments": list(TREATMENT_EFFECTS.keys()),
        "effects":    TREATMENT_EFFECTS
    }


@router.post(
    "/ranked",
    summary="Rank all treatments for a patient",
    description=(
        "Runs all treatments through the weighted scoring engine and returns "
        "them ranked best to worst. Results are saved to the database."
    )
)
def get_ranked_treatments(request: CompareRequest, db: Session = Depends(get_db)):
    try:
        vitals = parse_vitals(request.vitals_history)
        ranked = score_and_rank(vitals)
        best   = ranked[0]

        result = SimulationResult(
            patient_id         = request.patient_id,
            best_treatment     = best["treatment"],
            baseline_severity  = best["baseline_severity"],
            projected_severity = best["projected_severity"],
            improvement        = best["improvement"],
            final_score        = best["final_score"],
            full_results       = json.dumps(ranked, default=str)
        )
        db.add(result)
        db.commit()

        return {
            "total_treatments":  len(ranked),
            "best_treatment":    best["treatment"],
            "months_of_data":    len(request.vitals_history),
            "saved_to_db":       True,
            "ranked_treatments": ranked
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/quick-verdict",
    summary="Get single best treatment recommendation",
    description="Returns only the top-ranked treatment with a plain-English summary."
)
def quick_verdict(request: CompareRequest):
    try:
        vitals = parse_vitals(request.vitals_history)
        ranked = score_and_rank(vitals)
        best   = ranked[0]

        return {
            "recommended_treatment": best["treatment"],
            "final_score":           best["final_score"],
            "baseline_severity":     best["baseline_severity"],
            "baseline_risk":         best["baseline_risk"],
            "projected_severity":    best["projected_severity"],
            "projected_risk":        best["projected_risk"],
            "improvement":           best["improvement"],
            "summary": (
                f"{best['treatment']} is the best option. "
                f"Severity reduces from {best['baseline_severity']} "
                f"({best['baseline_risk']} risk) to "
                f"{best['projected_severity']} "
                f"({best['projected_risk']} risk) over 6 months."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    summary="Get all past simulation results",
    description="Returns every simulation that has been saved to the database, newest first."
)
def get_history(db: Session = Depends(get_db)):
    try:
        results = db.query(SimulationResult).order_by(
            SimulationResult.created_at.desc()
        ).all()

        return {
            "total": len(results),
            "simulations": [
                {
                    "id":                 r.id,
                    "patient_id":         r.patient_id,
                    "best_treatment":     r.best_treatment,
                    "baseline_severity":  r.baseline_severity,
                    "projected_severity": r.projected_severity,
                    "improvement":        r.improvement,
                    "final_score":        r.final_score,
                    "timestamp":          r.created_at
                }
                for r in results
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
