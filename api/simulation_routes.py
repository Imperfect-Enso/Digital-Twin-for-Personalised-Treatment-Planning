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
from db.database import get_db, PrognosisLog, PatientProfile, VitalHistory
from middleware.auth import verify_token


# Router
router = APIRouter(prefix="/patient", tags=["ML Simulation"])


# Request models

class VitalReading(BaseModel):
    blood_pressure: float = Field(..., ge=60.0, le=250.0, example=94.2)
    glucose: float        = Field(..., ge=40.0, le=500.0, example=112.4)
    heart_rate: float     = Field(..., ge=30.0, le=220.0, example=78.7)
    inflammation: float   = Field(..., ge=0.1, le=10.0, example=1.99)


class PrognosisRequest(BaseModel):
    patient_id: int | None = Field(None, example=1)
    vitals_history: list[VitalReading] = Field(..., min_length=3)


class SimulationRequest(BaseModel):
    vitals_history: list[VitalReading] = Field(..., min_length=3)
    treatment: str = Field(..., example="medication_A")


class CompareRequest(BaseModel):
    vitals_history: list[VitalReading] = Field(..., min_length=3)


# Patient schemas

class VitalEntry(BaseModel):
    timestep: int
    blood_pressure: float = Field(..., ge=60.0, le=250.0)
    glucose: float        = Field(..., ge=40.0, le=500.0)
    heart_rate: float     = Field(..., ge=30.0, le=220.0)
    inflammation: float   = Field(..., ge=0.1, le=10.0)


class VitalUpdateBody(BaseModel):
    blood_pressure: float = Field(..., ge=60.0, le=250.0)
    glucose: float        = Field(..., ge=40.0, le=500.0)
    heart_rate: float     = Field(..., ge=30.0, le=220.0)
    inflammation: float   = Field(..., ge=0.1, le=10.0)


class RegisterPatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    med_condition: str
    weight: float
    height: float
    vitals: list[VitalEntry] = []


class UpdatePatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    med_condition: str
    weight: float
    height: float


# Helper

def parse_vitals(vitals: list[VitalReading]) -> list[dict]:
    return [v.model_dump() for v in vitals]


# Get treatments

@router.get("/treatments")
def get_available_treatments(user: str = Depends(verify_token)):
    return {
        "available_treatments": list(TREATMENT_EFFECTS.keys()),
        "display_names": TREATMENT_NAMES,
        "descriptions": TREATMENT_DESCRIPTIONS,
        "effects": TREATMENT_EFFECTS
    }


# Prognosis

@router.post("/prognosis")
def get_prognosis(
    request: PrognosisRequest,
    db: Session = Depends(get_db),
    user: str = Depends(verify_token)
):
    try:
        vitals = parse_vitals(request.vitals_history)

        history = np.array(
            [[v[vital] for vital in VITALS] for v in vitals],
            dtype=np.float32
        )

        score = predict_severity(history)

        if score > 0.6:
            risk = "High"
            interpretation = "Patient shows high disease severity. Immediate treatment recommended."
        elif score > 0.3:
            risk = "Medium"
            interpretation = "Patient shows moderate severity. Close monitoring advised."
        else:
            risk = "Low"
            interpretation = "Patient shows low severity. Maintain current health practices."

        log = PrognosisLog(
            patient_id=request.patient_id,
            severity_score=score,
            risk_level=risk,
            interpretation=interpretation,
            months_of_data=len(request.vitals_history)
        )

        db.add(log)
        db.commit()

        return {
            "severity_score": score,
            "risk_level": risk,
            "interpretation": interpretation,
            "months_of_data": len(request.vitals_history),
            "saved_to_db": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Simulate treatment

@router.post("/simulate-treatment")
def simulate(
    request: SimulationRequest,
    user: str = Depends(verify_token)
):
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


# Compare treatments

@router.post("/compare-treatments")
def compare(
    request: CompareRequest,
    user: str = Depends(verify_token)
):
    try:
        vitals = parse_vitals(request.vitals_history)
        results = compare_all_treatments(vitals)
        return {"ranked_treatments": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Register patient

@router.post("/register")
def register_patient(
    request: RegisterPatientRequest,
    db: Session = Depends(get_db),
    user: str = Depends(verify_token)
):
    try:
        last = db.query(PatientProfile).order_by(
            PatientProfile.patient_id.desc()
        ).first()

        new_id = (last.patient_id + 1) if last else 1

        profile = PatientProfile(
            patient_id=new_id,
            name=request.name,
            age=request.age,
            gender=request.gender.upper(),
            med_condition=request.med_condition,
            weight=request.weight,
            height=request.height,
        )
        db.add(profile)

        for v in request.vitals:
            db.add(VitalHistory(
                patient_id=new_id,
                timestep=v.timestep,
                blood_pressure=v.blood_pressure,
                glucose=v.glucose,
                heart_rate=v.heart_rate,
                inflammation=v.inflammation,
            ))

        db.commit()
        db.refresh(profile)

        return {
            "success": True,
            "patient_id": new_id,
            "name": request.name,
            "message": f"Patient '{request.name}' registered successfully with ID {new_id}"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# IMPORTANT: keep this above /{patient_id} routes

@router.get("/list")
def list_patients(
    db: Session = Depends(get_db),
    user: str = Depends(verify_token)
):
    try:
        patients = db.query(PatientProfile).order_by(PatientProfile.patient_id).all()

        return {
            "total": len(patients),
            "patients": [
                {
                    "patient_id": p.patient_id,
                    "name": p.name,
                    "age": p.age,
                    "gender": p.gender,
                    "med_condition": p.med_condition,
                }
                for p in patients
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get profile

@router.get("/{patient_id}/profile")
def get_patient_profile(
    patient_id: int,
    db: Session = Depends(get_db),
    user: str = Depends(verify_token)
):
    try:
        profile = db.query(PatientProfile).filter(
            PatientProfile.patient_id == patient_id
        ).first()

        if not profile:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

        vitals = db.query(VitalHistory).filter(
            VitalHistory.patient_id == patient_id
        ).order_by(VitalHistory.timestep).all()

        return {
            "patient_id": profile.patient_id,
            "name": profile.name,
            "age": profile.age,
            "gender": profile.gender,
            "med_condition": profile.med_condition,
            "weight": profile.weight,
            "height": profile.height,
            "vital_history": [
                {
                    "timestep": v.timestep,
                    "blood_pressure": v.blood_pressure,
                    "glucose": v.glucose,
                    "heart_rate": v.heart_rate,
                    "inflammation": v.inflammation
                }
                for v in vitals
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))