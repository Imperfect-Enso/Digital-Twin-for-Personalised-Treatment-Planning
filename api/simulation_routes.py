from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import numpy as np

from ml.simulator import (
    simulate_treatment,
    compare_all_treatments,
    predict_severity,
    TREATMENT_EFFECTS,
    VITALS
)
from ml.treatment_config import TREATMENT_NAMES, TREATMENT_DESCRIPTIONS
from db.database import get_db, PrognosisLog

router = APIRouter(prefix="/patient", tags=["ML Simulation"])


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------

class VitalReading(BaseModel):
    blood_pressure: float = Field(..., example=94.2)
    glucose:        float = Field(..., example=112.4)
    heart_rate:     float = Field(..., example=78.7)
    inflammation:   float = Field(..., example=1.99)


class PrognosisRequest(BaseModel):
    patient_id:     int | None = Field(None, example=1)
    vitals_history: list[VitalReading] = Field(..., min_length=3)


class SimulationRequest(BaseModel):
    vitals_history: list[VitalReading] = Field(..., min_length=3)
    treatment:      str = Field(..., example="medication_A")


class CompareRequest(BaseModel):
    vitals_history: list[VitalReading] = Field(..., min_length=3)


def parse_vitals(vitals: list[VitalReading]) -> list[dict]:
    return [v.model_dump() for v in vitals]


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.get("/treatments")
def get_available_treatments():
    return {
        "available_treatments": list(TREATMENT_EFFECTS.keys()),
        "display_names":        TREATMENT_NAMES,
        "descriptions":         TREATMENT_DESCRIPTIONS,
        "effects":              TREATMENT_EFFECTS
    }


@router.post("/prognosis")
def get_prognosis(request: PrognosisRequest, db: Session = Depends(get_db)):
    try:
        vitals  = parse_vitals(request.vitals_history)
        history = np.array(
            [[v[vital] for vital in VITALS] for v in vitals],
            dtype=np.float32
        )
        score = predict_severity(history)

        if score > 0.6:
            risk           = "High"
            interpretation = "Patient shows high disease severity. Immediate treatment recommended."
        elif score > 0.3:
            risk           = "Medium"
            interpretation = "Patient shows moderate severity. Close monitoring advised."
        else:
            risk           = "Low"
            interpretation = "Patient shows low severity. Maintain current health practices."

        log = PrognosisLog(
            patient_id     = request.patient_id,
            severity_score = score,
            risk_level     = risk,
            interpretation = interpretation,
            months_of_data = len(request.vitals_history)
        )
        db.add(log)
        db.commit()

        return {
            "severity_score": score,
            "risk_level":     risk,
            "interpretation": interpretation,
            "months_of_data": len(request.vitals_history),
            "saved_to_db":    True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate-treatment")
def simulate(request: SimulationRequest):
    try:
        if request.treatment not in TREATMENT_EFFECTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown treatment. Valid options: {list(TREATMENT_EFFECTS.keys())}"
            )
        vitals = parse_vitals(request.vitals_history)
        return simulate_treatment(vitals, request.treatment)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-treatments")
def compare(request: CompareRequest):
    try:
        vitals  = parse_vitals(request.vitals_history)
        results = compare_all_treatments(vitals)
        return {"ranked_treatments": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{patient_id}/profile",
    summary="Get patient profile and vital history",
    description="Returns full patient info and 12-month vital history for any of the 10 preset patients."
)
def get_patient_profile(patient_id: int, db: Session = Depends(get_db)):
    try:
        from db.database import PatientProfile, VitalHistory

        profile = db.query(PatientProfile).filter(
            PatientProfile.patient_id == patient_id
        ).first()

        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Patient {patient_id} not found. Valid IDs are 1-10."
            )

        vitals = db.query(VitalHistory).filter(
            VitalHistory.patient_id == patient_id
        ).order_by(VitalHistory.timestep).all()

        return {
            "patient_id":    profile.patient_id,
            "name":          profile.name,
            "age":           profile.age,
            "gender":        profile.gender,
            "med_condition": profile.med_condition,
            "weight":        profile.weight,
            "height":        profile.height,
            "vital_history": [
                {
                    "timestep":       v.timestep,
                    "blood_pressure": v.blood_pressure,
                    "glucose":        v.glucose,
                    "heart_rate":     v.heart_rate,
                    "inflammation":   v.inflammation
                }
                for v in vitals
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{patient_id}/progression",
    summary="Get no-treatment disease progression timeline",
    description=(
        "Returns severity scores at 4 time intervals showing projected deterioration "
        "without intervention. Used for the progression timeline in the frontend."
    )
)
def get_progression(patient_id: int, db: Session = Depends(get_db)):
    try:
        from db.database import VitalHistory
        import numpy as np

        vitals_rows = db.query(VitalHistory).filter(
            VitalHistory.patient_id == patient_id
        ).order_by(VitalHistory.timestep).all()

        if not vitals_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No vitals found for patient {patient_id}"
            )

        vitals = [
            {
                "blood_pressure": v.blood_pressure,
                "glucose":        v.glucose,
                "heart_rate":     v.heart_rate,
                "inflammation":   v.inflammation
            }
            for v in vitals_rows
        ]

        no_treatment   = simulate_treatment(vitals, "no_treatment")
        projected      = no_treatment["projected_vitals"]
        baseline_score = no_treatment["baseline_severity"]

        def severity_label(score):
            if score > 0.6:
                return "High"
            elif score > 0.3:
                return "Medium"
            else:
                return "Low"

        stages = []
        for i, (label, months) in enumerate([
            ("M1-3",  projected[0:3]),
            ("M3-6",  projected[3:6]),
            ("M6-9",  projected[3:6]),
            ("M9-12", projected[3:6]),
        ]):
            last_vital = months[-1]
            deviation  = (
                (last_vital["blood_pressure"] - 80)  / 30 +
                (last_vital["glucose"]        - 90)  / 50 +
                (last_vital["inflammation"]   - 1.0) / 0.8
            ) / 3
            score = round(min(max(baseline_score + (deviation * 0.1 * (i + 1)), 0), 1), 4)

            stages.append({
                "stage":          label,
                "severity_score": score,
                "risk_level":     severity_label(score),
                "avg_bp":         round(sum(v["blood_pressure"] for v in months) / len(months), 2),
                "avg_glucose":    round(sum(v["glucose"]        for v in months) / len(months), 2),
                "avg_hr":         round(sum(v["heart_rate"]     for v in months) / len(months), 2),
                "avg_inflam":     round(sum(v["inflammation"]   for v in months) / len(months), 4),
            })

        return {
            "patient_id":     patient_id,
            "baseline_score": baseline_score,
            "baseline_risk":  severity_label(baseline_score),
            "scenario":       "no_treatment",
            "message":        "Projected deterioration without any intervention",
            "timeline":       stages
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/list",
    summary="Get all 10 preset patients",
    description="Returns the patient list used to populate the frontend selector."
)
def list_patients(db: Session = Depends(get_db)):
    try:
        from db.database import PatientProfile

        patients = db.query(PatientProfile).order_by(PatientProfile.patient_id).all()

        return {
            "total": len(patients),
            "patients": [
                {
                    "patient_id":    p.patient_id,
                    "name":          p.name,
                    "age":           p.age,
                    "gender":        p.gender,
                    "med_condition": p.med_condition,
                }
                for p in patients
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
