import json
import numpy as np
import pandas as pd
from pathlib import Path


INPUT_CSV       = Path(__file__).resolve().parent / "patients.csv"
OUTPUT_CSV      = Path(__file__).resolve().parent / "patients.csv"
TARGET_PATIENTS = 200
NOISE_LEVEL     = 0.04  # 4% gaussian noise per vital reading

VITALS = ["blood_pressure", "glucose", "heart_rate", "inflammation"]

# Augmented values are clamped to these ranges to stay within model training bounds
CLAMP = {
    "blood_pressure": (80.0,  105.0),
    "glucose":        (85.0,  135.0),
    "heart_rate":     (65.0,  90.0),
    "inflammation":   (0.90,  1.75),
}


def augment_patient(source_rows: pd.DataFrame, new_patient_id: int) -> list[dict]:
    """
    Creates one synthetic patient by adding small gaussian noise to a real
    patient's 12-month vital sequence. Disease trend is preserved.
    """
    rows = []

    for _, visit in source_rows.iterrows():
        new_row = {
            "patient_id":     new_patient_id,
            "timestep":       int(visit["timestep"]),
            "severity_label": int(visit["severity_label"])
        }

        for vital in VITALS:
            original = float(visit[vital])
            noise    = np.random.normal(0, original * NOISE_LEVEL)
            noisy    = original + noise
            lo, hi   = CLAMP[vital]
            clamped  = float(np.clip(noisy, lo, hi))
            new_row[vital] = round(clamped, 4)

        rows.append(new_row)

    return rows


def augment():
    """
    Expands the real patient dataset to TARGET_PATIENTS by generating synthetic
    copies of existing patients with small random noise applied to vitals.
    Overwrites patients.csv with the combined real + synthetic dataset.
    """
    df_real          = pd.read_csv(INPUT_CSV)
    real_patient_ids = df_real["patient_id"].unique()
    num_real         = len(real_patient_ids)

    print(f"Real patients loaded:     {num_real}")
    print(f"Target after augment:     {TARGET_PATIENTS}")
    print(f"Synthetic to generate:    {TARGET_PATIENTS - num_real}\n")

    to_generate = TARGET_PATIENTS - num_real
    per_patient = to_generate // num_real
    remainder   = to_generate  % num_real

    new_rows  = []
    next_id   = int(df_real["patient_id"].max()) + 1
    generated = 0

    for i, pid in enumerate(real_patient_ids):
        source   = df_real[df_real["patient_id"] == pid].sort_values("timestep")
        severity = int(source["severity_label"].iloc[0])
        count    = per_patient + (1 if i < remainder else 0)

        for _ in range(count):
            synthetic = augment_patient(source, next_id)
            new_rows.extend(synthetic)
            next_id   += 1
            generated += 1

        print(f"  Patient {pid:02d} ({'Severe' if severity == 1 else 'Mild  '}) "
              f"→ generated {count} synthetic copies")

    df_synthetic = pd.DataFrame(new_rows)
    df_final     = pd.concat([df_real, df_synthetic], ignore_index=True)
    df_final     = df_final.sort_values(["patient_id", "timestep"]).reset_index(drop=True)

    # Validation report
    severity_counts = dict(df_final.groupby("severity_label")["patient_id"].nunique())
    print(f"\nValidation:")
    print(f"  Total patients:   {df_final['patient_id'].nunique()}")
    print(f"  Total rows:       {len(df_final)}")
    print(f"  Severity balance: {severity_counts}")
    print(f"  Inflammation max: {df_final['inflammation'].max():.4f}  (must be < 1.81)")
    print(f"  Glucose max:      {df_final['glucose'].max():.4f}    (must be < 140)")
    print(f"  BP max:           {df_final['blood_pressure'].max():.4f}   (must be < 110)")
    print(f"  Missing values:   {df_final.isnull().sum().sum()}")

    mild_count   = severity_counts.get(0, 0)
    severe_count = severity_counts.get(1, 0)
    ratio        = min(mild_count, severe_count) / max(mild_count, severe_count)

    if ratio < 0.7:
        print(f"\n  WARNING: Dataset is imbalanced ({mild_count} mild vs {severe_count} severe)")
    else:
        print(f"\n  Dataset balanced and within safe ranges — ready for training!")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")
    print(f"Total rows: {len(df_final)}")


if __name__ == "__main__":
    augment()
