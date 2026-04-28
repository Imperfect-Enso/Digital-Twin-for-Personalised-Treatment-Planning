import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.simulator import (
    predict_severity,
    project_vitals,
    simulate_treatment,
    compare_all_treatments,
    VITALS,
    HEALTHY_BASELINE,
    TREATMENT_EFFECTS,
)


# Sample data

HEALTHY_VITALS = [
    {"blood_pressure": 120.3, "glucose": 83.8, "heart_rate": 162.5, "inflammation": 1.2273},
    {"blood_pressure": 120.2, "glucose": 83.7, "heart_rate": 163.8, "inflammation": 1.232},
    {"blood_pressure": 119.9, "glucose": 84.0, "heart_rate": 163.4, "inflammation": 1.2291},
    {"blood_pressure": 120.0, "glucose": 83.2, "heart_rate": 162.0, "inflammation": 1.2257},
    {"blood_pressure": 119.4, "glucose": 83.3, "heart_rate": 161.3, "inflammation": 1.217},
    {"blood_pressure": 120.3, "glucose": 83.2, "heart_rate": 161.3, "inflammation": 1.2083},
    {"blood_pressure": 120.0, "glucose": 83.3, "heart_rate": 160.4, "inflammation": 1.2106},
    {"blood_pressure": 119.6, "glucose": 83.1, "heart_rate": 159.9, "inflammation": 1.2218},
    {"blood_pressure": 119.6, "glucose": 82.7, "heart_rate": 160.6, "inflammation": 1.2144},
    {"blood_pressure": 119.7, "glucose": 81.9, "heart_rate": 159.5, "inflammation": 1.2156},
    {"blood_pressure": 120.2, "glucose": 81.9, "heart_rate": 159.4, "inflammation": 1.2137},
    {"blood_pressure": 119.3, "glucose": 81.7, "heart_rate": 159.1, "inflammation": 1.2201},
]

SICK_VITALS = [
    {"blood_pressure": 124.9, "glucose": 139.7, "heart_rate": 132.7, "inflammation": 1.2903},
    {"blood_pressure": 126.8, "glucose": 138.9, "heart_rate": 130.9, "inflammation": 1.3048},
    {"blood_pressure": 126.6, "glucose": 140.8, "heart_rate": 131.0, "inflammation": 1.3201},
    {"blood_pressure": 126.1, "glucose": 142.1, "heart_rate": 133.1, "inflammation": 1.3293},
    {"blood_pressure": 127.3, "glucose": 141.7, "heart_rate": 133.2, "inflammation": 1.3586},
    {"blood_pressure": 127.6, "glucose": 143.7, "heart_rate": 133.3, "inflammation": 1.3613},
    {"blood_pressure": 128.8, "glucose": 143.4, "heart_rate": 133.6, "inflammation": 1.3739},
    {"blood_pressure": 130.8, "glucose": 145.0, "heart_rate": 133.9, "inflammation": 1.3997},
    {"blood_pressure": 130.0, "glucose": 145.6, "heart_rate": 135.2, "inflammation": 1.4191},
    {"blood_pressure": 130.4, "glucose": 146.2, "heart_rate": 135.0, "inflammation": 1.4122},
    {"blood_pressure": 130.4, "glucose": 148.5, "heart_rate": 136.7, "inflammation": 1.4427},
    {"blood_pressure": 132.4, "glucose": 148.5, "heart_rate": 138.1, "inflammation": 1.4682},
]

SEVERE_VITALS = [
    {"blood_pressure": 168.0, "glucose": 93.0, "heart_rate": 160.0, "inflammation": 0.9106},
    {"blood_pressure": 170.4, "glucose": 93.2, "heart_rate": 159.5, "inflammation": 0.9144},
    {"blood_pressure": 169.2, "glucose": 95.0, "heart_rate": 161.7, "inflammation": 0.9203},
    {"blood_pressure": 169.7, "glucose": 95.3, "heart_rate": 161.1, "inflammation": 0.9398},
    {"blood_pressure": 170.7, "glucose": 94.9, "heart_rate": 161.7, "inflammation": 0.9446},
    {"blood_pressure": 172.5, "glucose": 96.1, "heart_rate": 161.2, "inflammation": 0.9544},
    {"blood_pressure": 173.9, "glucose": 96.6, "heart_rate": 163.2, "inflammation": 0.9602},
    {"blood_pressure": 173.3, "glucose": 97.5, "heart_rate": 163.4, "inflammation": 0.9729},
    {"blood_pressure": 176.8, "glucose": 97.5, "heart_rate": 164.5, "inflammation": 0.9879},
    {"blood_pressure": 178.1, "glucose": 98.9, "heart_rate": 163.8, "inflammation": 1.0033},
    {"blood_pressure": 177.7, "glucose": 99.2, "heart_rate": 163.7, "inflammation": 1.0127},
    {"blood_pressure": 178.9, "glucose": 98.2, "heart_rate": 164.0, "inflammation": 1.0096},
]


def vitals_to_array(vitals):
    return np.array(
        [[v[vital] for vital in VITALS] for v in vitals],
        dtype=np.float32
    )


# predict_severity

class TestPredictSeverity:

    def test_returns_float(self):
        score = predict_severity(vitals_to_array(HEALTHY_VITALS))
        assert isinstance(score, float)

    def test_in_range(self):
        for v in [HEALTHY_VITALS, SICK_VITALS, SEVERE_VITALS]:
            score = predict_severity(vitals_to_array(v))
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        arr = vitals_to_array(SICK_VITALS)
        assert predict_severity(arr) == predict_severity(arr)

    def test_severe_higher_than_healthy(self):
        assert predict_severity(vitals_to_array(SEVERE_VITALS)) > \
               predict_severity(vitals_to_array(HEALTHY_VITALS))

    def test_rounded(self):
        score = predict_severity(vitals_to_array(HEALTHY_VITALS))
        assert score == round(score, 4)


# project_vitals

class TestProjectVitals:

    def test_length(self):
        proj = project_vitals(vitals_to_array(HEALTHY_VITALS)[-1], "medication_A", 6)
        assert len(proj) == 6

    def test_keys_present(self):
        proj = project_vitals(vitals_to_array(HEALTHY_VITALS)[-1], "medication_A", 3)
        for step in proj:
            for v in VITALS:
                assert v in step

    def test_values_are_float(self):
        proj = project_vitals(vitals_to_array(SICK_VITALS)[-1], "lifestyle_changes", 6)
        for step in proj:
            for v in VITALS:
                assert isinstance(step[v], float)

    def test_medication_vs_no_treatment(self):
        last = vitals_to_array(SICK_VITALS)[-1]

        med = project_vitals(last, "medication_A", 6)
        none = project_vitals(last, "no_treatment", 6)

        avg_med = sum(p["inflammation"] for p in med) / 6
        avg_none = sum(p["inflammation"] for p in none) / 6

        assert avg_med < avg_none

    def test_no_treatment_worsens_bp(self):
        last = vitals_to_array(SICK_VITALS)[-1]
        start = last[VITALS.index("blood_pressure")]

        proj = project_vitals(last, "no_treatment", 6)
        assert proj[-1]["blood_pressure"] > start

    def test_all_treatments_supported(self):
        last = vitals_to_array(HEALTHY_VITALS)[-1]
        for t in TREATMENT_EFFECTS:
            assert len(project_vitals(last, t, 3)) == 3


# simulate_treatment

class TestSimulateTreatment:

    def test_required_keys(self):
        result = simulate_treatment(SICK_VITALS, "medication_A")
        for k in [
            "treatment", "display_name",
            "baseline_severity", "projected_severity",
            "baseline_risk", "projected_risk",
            "improvement", "projected_vitals", "recommendation"
        ]:
            assert k in result

    def test_improvement_correct(self):
        r = simulate_treatment(SICK_VITALS, "medication_A")
        assert abs(r["improvement"] - round(r["baseline_severity"] - r["projected_severity"], 4)) < 1e-6

    def test_risk_labels_valid(self):
        r = simulate_treatment(SICK_VITALS, "medication_B")
        assert r["baseline_risk"] in {"Low", "Medium", "High"}
        assert r["projected_risk"] in {"Low", "Medium", "High"}

    def test_recommendation_logic(self):
        for t in TREATMENT_EFFECTS:
            r = simulate_treatment(SICK_VITALS, t)
            assert (r["improvement"] > 0) == (r["recommendation"] == "Recommended")

    def test_projected_length(self):
        r = simulate_treatment(HEALTHY_VITALS, "lifestyle_changes")
        assert len(r["projected_vitals"]) == 6

    def test_invalid_treatment(self):
        with pytest.raises(ValueError):
            simulate_treatment(SICK_VITALS, "invalid")

    def test_min_data(self):
        with pytest.raises(ValueError):
            simulate_treatment(HEALTHY_VITALS[:2], "medication_A")

    def test_accepts_3_months(self):
        r = simulate_treatment(SICK_VITALS[:3], "medication_A")
        assert "baseline_severity" in r

    def test_display_name(self):
        r = simulate_treatment(HEALTHY_VITALS, "no_treatment")
        assert isinstance(r["display_name"], str)


# compare_all_treatments

class TestCompareAllTreatments:

    def test_all_returned(self):
        results = compare_all_treatments(SICK_VITALS)
        assert len(results) == len(TREATMENT_EFFECTS)

    def test_all_names_present(self):
        results = compare_all_treatments(SICK_VITALS)
        assert {r["treatment"] for r in results} == set(TREATMENT_EFFECTS.keys())

    def test_sorted(self):
        results = compare_all_treatments(SICK_VITALS)
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks(self):
        results = compare_all_treatments(SICK_VITALS)
        assert [r["rank"] for r in results] == list(range(1, len(results) + 1))

    def test_no_treatment_not_first(self):
        results = compare_all_treatments(SICK_VITALS)
        assert results[0]["treatment"] != "no_treatment"

    def test_extra_fields_present(self):
        for r in compare_all_treatments(SICK_VITALS):
            assert "vital_improvement" in r
            assert "speed_score" in r

    def test_final_score_type(self):
        for r in compare_all_treatments(HEALTHY_VITALS):
            assert isinstance(r["final_score"], float)

    def test_severe_prefers_medication(self):
        results = compare_all_treatments(SEVERE_VITALS)
        ranks = {r["treatment"]: r["rank"] for r in results}

        med_rank = min(
            ranks.get("medication_A", 99),
            ranks.get("medication_B", 99)
        )
        lifestyle_rank = ranks.get("lifestyle_changes", 99)

        assert med_rank < lifestyle_rank