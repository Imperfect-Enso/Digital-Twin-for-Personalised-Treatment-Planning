import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path

from ml.treatment_config import TREATMENT_NAMES, TREATMENT_DESCRIPTIONS


BASE_DIR = Path(__file__).resolve().parent.parent

model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_model.keras")

with open(BASE_DIR / "models" / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

VITALS = ["blood_pressure", "glucose", "heart_rate", "inflammation"]

# Per-vital effect multipliers applied each projected month.
# Negative values indicate improvement; 0.00 means no change.
TREATMENT_EFFECTS = {
    "medication_A": {
        "blood_pressure": -0.10,
        "glucose":        -0.05,
        "heart_rate":     -0.05,
        "inflammation":   -0.15
    },
    "medication_B": {
        "blood_pressure": -0.05,
        "glucose":        -0.20,
        "heart_rate":      0.00,
        "inflammation":   -0.10
    },
    "lifestyle_changes": {
        "blood_pressure": -0.08,
        "glucose":        -0.12,
        "heart_rate":     -0.10,
        "inflammation":   -0.08
    },
    "no_treatment": {
        "blood_pressure":  0.00,
        "glucose":         0.00,
        "heart_rate":      0.00,
        "inflammation":    0.00
    }
}

# Reference point for a healthy patient at baseline
HEALTHY_BASELINE = {
    "blood_pressure": 80.0,
    "glucose":        90.0,
    "heart_rate":     70.0,
    "inflammation":   1.0
}

# Maximum deviation observed in training data (severe patient, month 11).
# Used to normalise the vital deviation score to 0.0–1.0.
MAX_DEVIATION = {
    "blood_pressure": 30.0,
    "glucose":        45.0,
    "heart_rate":     15.0,
    "inflammation":    3.0
}


def predict_severity(vitals_sequence: np.ndarray) -> float:
    """
    Returns a severity score between 0.0 (healthy) and 1.0 (critical).

    Combines two signals:
      - 70% LSTM score  — captures learned temporal progression patterns
      - 30% vital deviation score — measures absolute distance from healthy baseline
    """
    timesteps, features = vitals_sequence.shape
    flat       = vitals_sequence.reshape(-1, features)
    scaled     = scaler.transform(flat)
    scaled_3d  = scaled.reshape(1, timesteps, features)
    lstm_score = float(model.predict(scaled_3d, verbose=0)[0][0])

    # Deviation score uses the most recent 3 readings (recent trend carries more weight)
    recent     = vitals_sequence[-3:]
    deviations = []

    for reading in recent:
        for i, vital in enumerate(VITALS):
            deviation  = max(0, reading[i] - HEALTHY_BASELINE[vital])
            normalized = min(deviation / MAX_DEVIATION[vital], 1.0)
            deviations.append(normalized)

    vital_score  = float(np.mean(deviations))
    hybrid_score = (0.7 * lstm_score) + (0.3 * vital_score)
    hybrid_score = min(max(hybrid_score, 0.0), 1.0)

    return round(hybrid_score, 4)


def project_vitals(last_known_vitals: np.ndarray, treatment_name: str,
                   future_steps: int = 6) -> list[dict]:
    """
    Applies treatment effects month-by-month from the last known reading.
    Returns a list of projected vital dicts, one per future month.
    """
    effects  = TREATMENT_EFFECTS[treatment_name]
    current  = last_known_vitals.copy().astype(float)
    projected = []

    for step in range(future_steps):
        next_vitals = {}

        for i, vital in enumerate(VITALS):
            if treatment_name == "no_treatment":
                # Natural disease progression — 2% worsening per month
                change = current[i] * 0.02
            else:
                change = current[i] * effects[vital]

            new_value = current[i] + change
            new_value = max(new_value, HEALTHY_BASELINE[vital])

            noise      = np.random.normal(0, 0.005 * new_value)
            current[i] = new_value + noise

            next_vitals[vital] = round(float(current[i]), 3)

        projected.append(next_vitals)

    return projected


def simulate_treatment(patient_vitals: list[dict], treatment_name: str,
                        future_steps: int = 6) -> dict:
    """
    Runs a single treatment simulation and returns severity scores,
    projected vitals, and a recommendation verdict.
    """
    if treatment_name not in TREATMENT_EFFECTS:
        raise ValueError(
            f"Unknown treatment '{treatment_name}'. "
            f"Choose from: {list(TREATMENT_EFFECTS.keys())}"
        )

    if len(patient_vitals) < 3:
        raise ValueError("Need at least 3 months of vitals history.")

    history = np.array(
        [[v[vital] for vital in VITALS] for v in patient_vitals],
        dtype=np.float32
    )

    baseline_severity = predict_severity(history)

    last_reading = history[-1]
    projected    = project_vitals(last_reading, treatment_name, future_steps)

    projected_array  = np.array(
        [[p[vital] for vital in VITALS] for p in projected],
        dtype=np.float32
    )
    full_sequence     = np.vstack([history, projected_array])
    prediction_window = full_sequence[-12:]

    projected_severity = predict_severity(prediction_window)
    improvement        = round(baseline_severity - projected_severity, 4)

    def severity_label(score):
        if score > 0.6:
            return "High"
        elif score > 0.3:
            return "Medium"
        else:
            return "Low"

    return {
        "treatment":          treatment_name,
        "display_name":       TREATMENT_NAMES.get(treatment_name, treatment_name),
        "baseline_severity":  baseline_severity,
        "projected_severity": projected_severity,
        "baseline_risk":      severity_label(baseline_severity),
        "projected_risk":     severity_label(projected_severity),
        "improvement":        improvement,
        "projected_vitals":   projected,
        "recommendation":     "Recommended" if improvement > 0 else "Not Recommended"
    }


def compare_all_treatments(patient_vitals: list[dict]) -> list[dict]:
    """
    Simulates all treatments, scores each one, and returns them
    ranked best to worst by final_score with rank numbers assigned.
    """
    results = []

    for treatment_name in TREATMENT_EFFECTS:
        result = simulate_treatment(patient_vitals, treatment_name)
        effects = TREATMENT_EFFECTS[treatment_name]

        # Average absolute improvement across all vitals with negative effects
        vital_improvement = sum(abs(v) for v in effects.values() if v < 0) / len(effects)

        # Proxy for how fast the treatment acts, based on total effect magnitude
        speed_score = min(1.0, sum(abs(v) for v in effects.values()) * 2)

        final_score = (
            result["improvement"] * 0.5 +
            vital_improvement      * 0.3 +
            speed_score            * 0.2
        )

        # Ensure no_treatment never outranks active treatments
        if treatment_name == "no_treatment":
            final_score = min(final_score, 0.0)

        result["vital_improvement"] = round(vital_improvement, 4)
        result["speed_score"]       = round(speed_score, 4)
        result["final_score"]       = round(final_score, 4)
        results.append(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# -----------------------------------------------------------------------------
# Quick Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    test_patient = [
        {"blood_pressure": 80.1,  "glucose": 91.2,  "heart_rate": 70.5, "inflammation": 1.02},
        {"blood_pressure": 80.8,  "glucose": 92.1,  "heart_rate": 71.1, "inflammation": 1.08},
        {"blood_pressure": 81.9,  "glucose": 93.8,  "heart_rate": 71.8, "inflammation": 1.15},
        {"blood_pressure": 83.2,  "glucose": 95.9,  "heart_rate": 72.6, "inflammation": 1.24},
        {"blood_pressure": 84.8,  "glucose": 98.4,  "heart_rate": 73.5, "inflammation": 1.35},
        {"blood_pressure": 86.7,  "glucose": 101.3, "heart_rate": 74.6, "inflammation": 1.48},
        {"blood_pressure": 88.9,  "glucose": 104.6, "heart_rate": 75.8, "inflammation": 1.63},
        {"blood_pressure": 91.4,  "glucose": 108.3, "heart_rate": 77.2, "inflammation": 1.80},
        {"blood_pressure": 94.2,  "glucose": 112.4, "heart_rate": 78.7, "inflammation": 1.99},
        {"blood_pressure": 97.3,  "glucose": 116.9, "heart_rate": 80.4, "inflammation": 2.20},
        {"blood_pressure": 100.7, "glucose": 121.8, "heart_rate": 82.2, "inflammation": 2.43},
        {"blood_pressure": 104.4, "glucose": 127.1, "heart_rate": 84.2, "inflammation": 2.68},
    ]

    print("=" * 60)
    print("TREATMENT COMPARISON FOR TEST PATIENT")
    print("=" * 60)

    results = compare_all_treatments(test_patient)

    for rank, result in enumerate(results, 1):
        print(f"\nRank #{rank}: {result['treatment'].upper()}")
        print(f"  Baseline severity:  {result['baseline_severity']} ({result['baseline_risk']} risk)")
        print(f"  Projected severity: {result['projected_severity']} ({result['projected_risk']} risk)")
        print(f"  Improvement:        {result['improvement']}")
        print(f"  Recommendation:     {result['recommendation']}")
