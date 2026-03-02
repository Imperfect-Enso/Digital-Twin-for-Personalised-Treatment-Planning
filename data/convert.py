import json
import numpy as np
import pandas as pd
from pathlib import Path


INPUT_DIR  = Path(__file__).resolve().parent / "raw"
OUTPUT_CSV = Path(__file__).resolve().parent / "patients.csv"

VITALS = ["blood_pressure", "glucose", "heart_rate", "inflammation"]

# Maps raw input ranges to the ML model's expected training ranges via min-max scaling
SCALE_CONFIG = {
    "blood_pressure": {"in_min": 110.0, "in_max": 135.0, "out_min": 80.0,  "out_max": 105.0},
    "glucose":        {"in_min": 150.0, "in_max": 200.0, "out_min": 85.0,  "out_max": 135.0},
    "heart_rate":     {"in_min": 60.0,  "in_max": 95.0,  "out_min": 65.0,  "out_max": 90.0},
    "inflammation":   {"in_min": 1.0,   "in_max": 2.0,   "out_min": 0.90,  "out_max": 1.75},
}


def scale_value(value: float, vital: str) -> float:
    """Scales a single vital reading from input range to model-expected range."""
    cfg    = SCALE_CONFIG[vital]
    ratio  = (value - cfg["in_min"]) / (cfg["in_max"] - cfg["in_min"])
    scaled = cfg["out_min"] + ratio * (cfg["out_max"] - cfg["out_min"])
    return round(float(np.clip(scaled, cfg["out_min"], cfg["out_max"])), 4)


def assign_severity(visits: list[dict]) -> int:
    """
    Infers severity label (0 = mild, 1 = severe) from the patient's vital trend.
    Compares average inflammation and glucose in the first 3 vs last 3 visits.
    A 5% or greater worsening in either vital classifies the patient as severe.
    """
    early_inflam  = sum(v["inflammation"] for v in visits[:3])  / 3
    late_inflam   = sum(v["inflammation"] for v in visits[-3:]) / 3
    early_glucose = sum(v["glucose"]      for v in visits[:3])  / 3
    late_glucose  = sum(v["glucose"]      for v in visits[-3:]) / 3

    inflammation_worsening = late_inflam  > early_inflam  * 1.05
    glucose_worsening      = late_glucose > early_glucose * 1.05

    return 1 if (inflammation_worsening or glucose_worsening) else 0


def convert_patient(filepath: Path) -> list[dict]:
    """
    Reads one raw patient .txt file and returns scaled, labelled row dicts.
    Raises ValueError if the file does not contain exactly 12 visits.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    patient_id = data["patient_id"]
    visits     = data["visit_history"]

    if len(visits) != 12:
        raise ValueError(f"Patient {patient_id} has {len(visits)} visits — expected 12.")

    severity = assign_severity(visits)

    rows = []
    for timestep, visit in enumerate(visits):
        row = {
            "patient_id":     patient_id,
            "timestep":       timestep,
            "blood_pressure": scale_value(visit["blood_pressure"], "blood_pressure"),
            "glucose":        scale_value(visit["glucose"],        "glucose"),
            "heart_rate":     scale_value(visit["heart_rate"],     "heart_rate"),
            "inflammation":   scale_value(visit["inflammation"],   "inflammation"),
            "severity_label": severity
        }
        rows.append(row)

    return rows


def convert_all():
    """
    Converts all .txt files in the raw/ folder to a single patients.csv.
    Prints a validation report on completion.
    """
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No txt files found in {INPUT_DIR}\n"
            f"Place raw patient files in: {INPUT_DIR}"
        )

    print(f"Found {len(txt_files)} patient files\n")

    all_rows = []

    for filepath in txt_files:
        try:
            rows     = convert_patient(filepath)
            severity = rows[0]["severity_label"]
            pid      = rows[0]["patient_id"]
            print(f"  Patient {pid:02d} → "
                  f"{'Severe' if severity == 1 else 'Mild  '} "
                  f"({len(rows)} rows) — {filepath.name}")
            all_rows.extend(rows)

        except Exception as e:
            print(f"  Failed to convert {filepath.name}: {e}")

    df = pd.DataFrame(all_rows)

    # Validation report
    print(f"\nValidation:")
    print(f"  Total rows:       {len(df)}  (expected {len(txt_files) * 12})")
    print(f"  Unique patients:  {df['patient_id'].nunique()}")
    print(f"  Severity balance: {dict(df.groupby('severity_label')['patient_id'].nunique())}")
    print(f"  Inflammation max: {df['inflammation'].max():.4f}  (must be < 1.81)")
    print(f"  Glucose max:      {df['glucose'].max():.2f}    (must be < 140)")
    print(f"  BP max:           {df['blood_pressure'].max():.2f}   (must be < 110)")
    print(f"  Missing values:   {df.isnull().sum().sum()}")

    if df["inflammation"].max() > 1.81:
        print("  WARNING: inflammation exceeds 1.81 — model may saturate!")
    if df["glucose"].max() > 140:
        print("  WARNING: glucose exceeds 140 — model may saturate!")
    else:
        print(f"\n  All values within safe range — ready for training!")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")

    return df


if __name__ == "__main__":
    convert_all()
