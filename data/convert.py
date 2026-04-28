import json
import numpy as np
import pandas as pd
from pathlib import Path


INPUT_DIR  = Path(__file__).resolve().parent / "raw"
OUTPUT_CSV = Path(__file__).resolve().parent / "patients.csv"

VITALS = ["blood_pressure", "glucose", "heart_rate", "inflammation"]

# scaling config (raw → model range)
SCALE_CONFIG = {
    "blood_pressure": {"in_min": 110.0, "in_max": 135.0, "out_min": 80.0,  "out_max": 105.0},
    "glucose":        {"in_min": 150.0, "in_max": 200.0, "out_min": 85.0,  "out_max": 135.0},
    "heart_rate":     {"in_min": 60.0,  "in_max": 95.0,  "out_min": 65.0,  "out_max": 90.0},
    "inflammation":   {"in_min": 1.0,   "in_max": 2.0,   "out_min": 0.90,  "out_max": 1.75},
}


# scale one value

def scale_value(value: float, vital: str) -> float:
    cfg = SCALE_CONFIG[vital]

    ratio = (value - cfg["in_min"]) / (cfg["in_max"] - cfg["in_min"])
    scaled = cfg["out_min"] + ratio * (cfg["out_max"] - cfg["out_min"])

    return round(float(np.clip(scaled, cfg["out_min"], cfg["out_max"])), 4)


# decide severity from trend

def assign_severity(visits: list[dict]) -> int:
    early_inf = sum(v["inflammation"] for v in visits[:3]) / 3
    late_inf  = sum(v["inflammation"] for v in visits[-3:]) / 3

    early_gl  = sum(v["glucose"] for v in visits[:3]) / 3
    late_gl   = sum(v["glucose"] for v in visits[-3:]) / 3

    inf_worse = late_inf > early_inf * 1.05
    gl_worse  = late_gl  > early_gl  * 1.05

    return 1 if (inf_worse or gl_worse) else 0


# convert one patient file

def convert_patient(filepath: Path) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)

    patient_id = data["patient_id"]
    visits = data["visit_history"]

    if len(visits) != 12:
        raise ValueError(f"Patient {patient_id} has {len(visits)} visits (expected 12)")

    severity = assign_severity(visits)

    rows = []
    for t, visit in enumerate(visits):
        rows.append({
            "patient_id": patient_id,
            "timestep": t,
            "blood_pressure": scale_value(visit["blood_pressure"], "blood_pressure"),
            "glucose": scale_value(visit["glucose"], "glucose"),
            "heart_rate": scale_value(visit["heart_rate"], "heart_rate"),
            "inflammation": scale_value(visit["inflammation"], "inflammation"),
            "severity_label": severity
        })

    return rows


# convert all files

def convert_all():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No txt files found in {INPUT_DIR}\n"
            f"Put raw files in: {INPUT_DIR}"
        )

    print(f"Found {len(txt_files)} files\n")

    all_rows = []

    for filepath in txt_files:
        try:
            rows = convert_patient(filepath)

            pid = rows[0]["patient_id"]
            severity = rows[0]["severity_label"]

            print(
                f"Patient {pid:02d} → "
                f"{'Severe' if severity == 1 else 'Mild'} "
                f"({len(rows)} rows)"
            )

            all_rows.extend(rows)

        except Exception as e:
            print(f"Failed: {filepath.name} → {e}")

    df = pd.DataFrame(all_rows)

    # quick validation

    print("\nValidation:")
    print(f"Total rows:       {len(df)}")
    print(f"Unique patients:  {df['patient_id'].nunique()}")
    print(f"Severity split:   {dict(df.groupby('severity_label')['patient_id'].nunique())}")
    print(f"Inflammation max: {df['inflammation'].max():.4f}")
    print(f"Glucose max:      {df['glucose'].max():.2f}")
    print(f"BP max:           {df['blood_pressure'].max():.2f}")
    print(f"Missing values:   {df.isnull().sum().sum()}")

    if df["inflammation"].max() > 1.81:
        print("WARNING: inflammation too high")
    if df["glucose"].max() > 140:
        print("WARNING: glucose too high")
    else:
        print("\nData looks fine")

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")

    return df


if __name__ == "__main__":
    convert_all()