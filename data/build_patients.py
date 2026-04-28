import numpy as np

np.random.seed(42)


# Load dataset

rows = []
with open("data/processed.cleveland.data") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 14 or "?" in parts:
            continue

        try:
            rows.append({
                "age": float(parts[0]),
                "sex": int(float(parts[1])),
                "trestbps": float(parts[3]),
                "chol": float(parts[4]),
                "thalach": float(parts[7]),
                "oldpeak": float(parts[9]),
                "target": int(float(parts[13]))
            })
        except:
            pass


# Select patients

mild_m   = sorted([r for r in rows if r["target"] == 0 and r["sex"] == 1], key=lambda x: x["chol"])
mild_f   = sorted([r for r in rows if r["target"] == 0 and r["sex"] == 0], key=lambda x: x["chol"])
severe_m = sorted([r for r in rows if r["target"] >= 1 and r["sex"] == 1], key=lambda x: x["chol"])
severe_f = sorted([r for r in rows if r["target"] >= 1 and r["sex"] == 0], key=lambda x: x["chol"])


SELECTED = [
    mild_m[10],
    mild_m[len(mild_m) // 2],
    mild_m[-10],
    mild_f[10],
    mild_f[len(mild_f) // 2],

    severe_m[10],
    severe_m[len(severe_m) // 2],
    severe_m[-10],
    severe_f[5],
    severe_f[-3],
]


# Names and conditions

NAMES = [
    "James Hooper",
    "David Keller",
    "Robert Simmons",
    "Emily Carter",
    "Susan Blake",
    "Michael Torres",
    "William Grant",
    "Thomas Nguyen",
    "Patricia Hughes",
    "Linda Ferrera",
]

CONDITIONS = {
    0: "No Heart Disease",
    1: "Mild Heart Disease",
    2: "Moderate Heart Disease",
    3: "Severe Heart Disease",
    4: "Critical Heart Disease",
}


# Mapping helpers

def map_blood_pressure(trestbps):
    return round(float(trestbps), 1)


def map_glucose(chol):
    return round(float(chol) / 2.3, 1)


def map_heart_rate(thalach):
    return round(float(thalach), 1)


def map_inflammation(oldpeak):
    val = 0.8 + float(oldpeak) * 0.22
    return round(max(0.8, min(val, 2.5)), 4)


# History generation

SEVERE_RATES = {
    "bp": 0.005,
    "glucose": 0.006,
    "hr": 0.003,
    "inflammation": 0.012,
}


def build_history(bp_final, gl_final, hr_final, inf_final, target):
    history = []

    if target >= 1:
        bp_0  = bp_final  / (1 + 11 * SEVERE_RATES["bp"])
        gl_0  = gl_final  / (1 + 11 * SEVERE_RATES["glucose"])
        hr_0  = hr_final  / (1 + 11 * SEVERE_RATES["hr"])
        inf_0 = inf_final / (1 + 11 * SEVERE_RATES["inflammation"])

        for t in range(12):
            progress = t / 11.0

            bp  = bp_0  + (bp_final  - bp_0)  * progress
            gl  = gl_0  + (gl_final  - gl_0)  * progress
            hr  = hr_0  + (hr_final  - hr_0)  * progress
            inf = inf_0 + (inf_final - inf_0) * progress

            bp  += np.random.normal(0, bp_final  * 0.005)
            gl  += np.random.normal(0, gl_final  * 0.005)
            hr  += np.random.normal(0, hr_final  * 0.005)
            inf += np.random.normal(0, inf_final * 0.005)

            history.append({
                "timestep": t,
                "blood_pressure": round(max(bp, 80.0), 1),
                "glucose": round(max(gl, 60.0), 1),
                "heart_rate": round(max(hr, 55.0), 1),
                "inflammation": round(max(inf, 0.8), 4),
            })

    else:
        bp_c, gl_c, hr_c, inf_c = bp_final, gl_final, hr_final, inf_final

        for t in range(12):
            bp_c  += np.random.normal(0, bp_c  * 0.005)
            gl_c  += np.random.normal(0, gl_c  * 0.005)
            hr_c  += np.random.normal(0, hr_c  * 0.005)
            inf_c += np.random.normal(0, inf_c * 0.005)

            history.append({
                "timestep": t,
                "blood_pressure": round(max(bp_c, 80.0), 1),
                "glucose": round(max(gl_c, 60.0), 1),
                "heart_rate": round(max(hr_c, 55.0), 1),
                "inflammation": round(max(inf_c, 0.8), 4),
            })

    return history


# Build patients

def build_preset_patients():
    patients = []

    for i, r in enumerate(SELECTED):
        bp_final  = map_blood_pressure(r["trestbps"])
        gl_final  = map_glucose(r["chol"])
        hr_final  = map_heart_rate(r["thalach"])
        inf_final = map_inflammation(r["oldpeak"])

        vitals = build_history(bp_final, gl_final, hr_final, inf_final, r["target"])

        patients.append({
            "patient_id": i + 1,
            "name": NAMES[i],
            "age": int(r["age"]),
            "gender": "M" if r["sex"] == 1 else "F",
            "med_condition": CONDITIONS[r["target"]],
            "weight": round(25.0 * (1.70 ** 2), 1),
            "height": 170.0,
            "vitals": vitals,
        })

    return patients


# Print output

def print_preset_patients(patients):
    print("PRESET_PATIENTS = [")

    for p in patients:
        print("    {")
        print(f"        \"patient_id\": {p['patient_id']}, \"name\": \"{p['name']}\", \"age\": {p['age']}, \"gender\": \"{p['gender']}\",")
        print(f"        \"med_condition\": \"{p['med_condition']}\", \"weight\": {p['weight']}, \"height\": {p['height']},")
        print("        \"vitals\": [")

        for v in p["vitals"]:
            print(
                f"            {{\"timestep\": {v['timestep']:>2}, "
                f"\"blood_pressure\": {v['blood_pressure']:>6}, "
                f"\"glucose\": {v['glucose']:>6}, "
                f"\"heart_rate\": {v['heart_rate']:>6}, "
                f"\"inflammation\": {v['inflammation']}}},"
            )

        print("        ]")
        print("    },")

    print("]")


if __name__ == "__main__":
    patients = build_preset_patients()
    print_preset_patients(patients)