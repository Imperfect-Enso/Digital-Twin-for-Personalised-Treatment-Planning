from ml.simulator import compare_all_treatments


# Wrapper to keep scoring logic consistent across endpoints

def score_and_rank(patient_vitals: list[dict]) -> list[dict]:
    return compare_all_treatments(patient_vitals)