import numpy as np
import pandas as pd
import os


def generate_patient_data(num_patients=200, timesteps=12):
    """
    Generates synthetic longitudinal patient health records for model training.

    Each patient is randomly assigned mild (0) or severe (1) status.
    Severe patients worsen progressively each month; mild patients remain stable.

    Args:
        num_patients: Number of synthetic patients to generate (default 200)
        timesteps:    Monthly readings per patient (default 12)

    Returns:
        DataFrame saved to data/patients.csv
    """
    records = []

    for patient_id in range(num_patients):
        base_severity = np.random.choice([0, 1])

        for t in range(timesteps):
            trend = base_severity * 0.03 * t
            noise = np.random.normal(0, 0.05, 4)

            record = {
                "patient_id":     patient_id,
                "timestep":       t,
                "blood_pressure": round(80  + trend * 10 + noise[0] * 10, 2),
                "glucose":        round(90  + trend * 15 + noise[1] * 15, 2),
                "heart_rate":     round(70  + trend * 5  + noise[2] * 8,  2),
                "inflammation":   round(1.0 + trend * 2  + noise[3],      4),
                "severity_label": base_severity
            }

            records.append(record)

    df = pd.DataFrame(records)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/patients.csv", index=False)

    print(f"Done! Generated {num_patients} patients x {timesteps} timesteps")
    print(f"Total rows: {len(df)}")
    print(f"\nSample (first patient):")
    print(df[df["patient_id"] == 0].to_string(index=False))

    return df


if __name__ == "__main__":
    generate_patient_data()
