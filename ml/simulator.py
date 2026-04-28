import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import threading

from ml.treatment_config import TREATMENT_NAMES, TREATMENT_DESCRIPTIONS


_predict_lock = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_model.keras")

with open(BASE_DIR / "models" / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


VITALS = ["blood_pressure", "glucose", "heart_rate", "inflammation"]


# Treatment effects

TREATMENT_EFFECTS = {
    "medication_A": {
        "blood_pressure": -0.15,
        "glucose": -0.03,
        "heart_rate": -0.08,
        "inflammation": -0.05
    },
    "medication_B": {
        "blood_pressure": -0.03,
        "glucose": -0.20,
        "heart_rate": 0.00,
        "inflammation": -0.10
    },
    "lifestyle_changes": {
        "blood_pressure": -0.06,
        "glucose": -0.10,
        "heart_rate": -0.10,
        "inflammation": -0.08
    },
    "no_treatment": {
        "blood_pressure": 0.00,
        "glucose": 0.00,
        "heart_rate": 0.00,
        "inflammation": 0.00
    }
}


# Baselines

HEALTHY_BASELINE = {
    "blood_pressure": 120.0,
    "glucose": 90.0,
    "heart_rate": 150.0,
    "inflammation": 1.0
}

MAX_DEVIATION = {
    "blood_pressure": 80.0,
    "glucose": 60.0,
    "heart_rate": 50.0,
    "inflammation": 1.5
}


# Predict severity

def predict_severity(vitals_sequence: np.ndarray) -> float:
    timesteps, features = vitals_sequence.shape

    flat = vitals_sequence.reshape(-1, features)
    scaled = scaler.transform(flat)
    scaled_3d = scaled.reshape(1, timesteps, features)

    with _predict_lock:
        lstm_score = float(model.predict(scaled_3d, verbose=0)[0][0])

    recent = vitals_sequence[-3:]
    deviations = []

    for reading in recent:
        for i, vital in enumerate(VITALS):
            deviation = max(0, reading[i] - HEALTHY_BASELINE[vital])
            normalized = min(deviation / MAX_DEVIATION[vital], 1.0)
            deviations.append(normalized)

    vital_score = float(np.mean(deviations))

    hybrid_score = (0.7 * lstm_score) + (0.3 * vital_score)
    hybrid_score = min(max(hybrid_score, 0.0), 1.0)

    return round(hybrid_score, 4)


# Project future vitals

def project_vitals(last_known_vitals: np.ndarray, treatment_name: str, future_steps: int = 6) -> list[dict]:
    effects = TREATMENT_EFFECTS[treatment_name]
    current = last_known_vitals.copy().astype(float)

    projected = []

    for _ in range(future_steps):
        next_vitals = {}

        for i, vital in enumerate(VITALS):
            if treatment_name == "no_treatment":
                change = current[i] * 0.02
            else:
                change = current[i] * effects[vital]

            new_value = current[i] + change
            new_value = max(new_value, HEALTHY_BASELINE[vital])

            noise = np.random.normal(0, 0.005 * new_value)
            current[i] = new_value + noise

            next_vitals[vital] = round(float(current[i]), 3)

        projected.append(next_vitals)

    return projected


# Simulate one treatment

def simulate_treatment(patient_vitals: list[dict], treatment_name: str, future_steps: int = 6) -> dict:
    if treatment_name not in TREATMENT_EFFECTS:
        raise ValueError(f"Unknown treatment '{treatment_name}'")

    if len(patient_vitals) < 3:
        raise ValueError("Need at least 3 months of data")

    history = np.array(
        [[v[vital] for vital in VITALS] for v in patient_vitals],
        dtype=np.float32
    )

    baseline_severity = predict_severity(history)

    projected = project_vitals(history[-1], treatment_name, future_steps)

    projected_array = np.array(
        [[p[vital] for vital in VITALS] for p in projected],
        dtype=np.float32
    )

    full_sequence = np.vstack([history, projected_array])
    prediction_window = full_sequence[-12:]

    projected_severity = predict_severity(prediction_window)
    improvement = round(baseline_severity - projected_severity, 4)

    def severity_label(score):
        if score > 0.6:
            return "High"
        elif score > 0.3:
            return "Medium"
        return "Low"

    return {
        "treatment": treatment_name,
        "display_name": TREATMENT_NAMES.get(treatment_name, treatment_name),
        "baseline_severity": baseline_severity,
        "projected_severity": projected_severity,
        "baseline_risk": severity_label(baseline_severity),
        "projected_risk": severity_label(projected_severity),
        "improvement": improvement,
        "projected_vitals": projected,
        "recommendation": "Recommended" if improvement > 0 else "Not Recommended"
    }


# Compare all treatments

def compare_all_treatments(patient_vitals: list[dict]) -> list[dict]:
    results = []

    last_reading = patient_vitals[-1]

    gaps = {
        vital: max(0.0, last_reading[vital] - HEALTHY_BASELINE[vital])
        for vital in VITALS
    }
    total_gap = sum(gaps.values())

    for treatment_name in TREATMENT_EFFECTS:
        result = simulate_treatment(patient_vitals, treatment_name)
        effects = TREATMENT_EFFECTS[treatment_name]

        if total_gap > 0:
            vital_improvement = sum(
                abs(effects[v]) * (gaps[v] / total_gap)
                for v in VITALS if effects[v] < 0
            )
        else:
            vital_improvement = 0.0

        speed_score = min(1.0, sum(abs(v) for v in effects.values()) * 2)

        final_score = (
            result["improvement"] * 0.5 +
            vital_improvement * 0.3 +
            speed_score * 0.2
        )

        if treatment_name == "no_treatment":
            final_score = min(final_score, 0.0)

        result["vital_improvement"] = round(vital_improvement, 4)
        result["speed_score"] = round(speed_score, 4)
        result["final_score"] = round(final_score, 4)

        results.append(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results