from ml.simulator import compare_all_treatments, VITALS


# Reference vitals for a healthy patient — used to measure gap closure
HEALTHY_BASELINE = {
    "blood_pressure": 80.0,
    "glucose":        90.0,
    "heart_rate":     70.0,
    "inflammation":   1.0
}


def compute_vital_improvement(baseline_vitals: list[dict],
                               projected_vitals: list[dict]) -> float:
    """
    Measures what percentage of each vital's gap from healthy baseline
    was closed by the treatment projection.

    Returns a score between 0.0 (no improvement) and 1.0 (fully normalised).
    Only vitals that were above healthy baseline are counted.
    """
    last_real = baseline_vitals[-1]
    last_proj = projected_vitals[-1]

    improvements = []

    for vital in VITALS:
        before_gap = last_real[vital] - HEALTHY_BASELINE[vital]
        after_gap  = last_proj[vital] - HEALTHY_BASELINE[vital]

        if before_gap > 0:
            pct_closed = (before_gap - after_gap) / before_gap
            improvements.append(max(0.0, min(pct_closed, 1.0)))

    if not improvements:
        return 0.0

    return round(sum(improvements) / len(improvements), 4)


def compute_treatment_speed(projected_vitals: list[dict]) -> float:
    """
    Measures how quickly improvement occurs across the 6 projected months.

    Compares average inflammation in the first half of the projection
    against the second half. A faster-acting treatment shows more
    improvement early. Returns a score between 0.0 (slow) and 1.0 (fast).
    """
    if len(projected_vitals) < 2:
        return 0.5

    midpoint    = len(projected_vitals) // 2
    first_half  = projected_vitals[:midpoint]
    second_half = projected_vitals[midpoint:]

    early_avg = sum(v["inflammation"] for v in first_half)  / len(first_half)
    late_avg  = sum(v["inflammation"] for v in second_half) / len(second_half)

    if early_avg == 0:
        return 0.5

    speed = (early_avg - late_avg) / early_avg

    return round(min(max(speed + 0.5, 0.0), 1.0), 4)


def score_and_rank(patient_vitals: list[dict]) -> list[dict]:
    """
    Runs all treatments through the simulator and scores each using
    a three-factor weighted formula:

        final_score = severity_improvement * 0.50
                    + vital_improvement    * 0.30
                    + treatment_speed      * 0.20

    Returns treatments sorted best to worst with rank numbers assigned.
    Called directly by comparison_routes.py.
    """
    raw_results = compare_all_treatments(patient_vitals)
    scored      = []

    for result in raw_results:
        sev_score   = max(0.0, result["improvement"])
        vital_score = compute_vital_improvement(patient_vitals, result["projected_vitals"])
        speed_score = compute_treatment_speed(result["projected_vitals"])

        final_score = round(
            (sev_score   * 0.50) +
            (vital_score * 0.30) +
            (speed_score * 0.20),
            4
        )

        scored.append({
            **result,
            "vital_improvement": vital_score,
            "speed_score":       speed_score,
            "final_score":       final_score
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    for i, result in enumerate(scored):
        result["rank"] = i + 1

    return scored


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
    print("WEIGHTED SCORING — TREATMENT COMPARISON")
    print("=" * 60)

    results = score_and_rank(test_patient)

    for r in results:
        print(f"\nRank #{r['rank']}: {r['treatment'].upper()}")
        print(f"  Severity improvement: {r['improvement']}")
        print(f"  Vital improvement:    {r['vital_improvement']}")
        print(f"  Speed score:          {r['speed_score']}")
        print(f"  Final score:          {r['final_score']}")
        print(f"  Recommendation:       {r['recommendation']}")
