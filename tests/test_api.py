import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


# Sample vitals (used across tests)

SAMPLE_VITALS = [
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


# Auth helpers

def get_token(username="doctor", password="password123") -> str:
    res = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )
    assert res.status_code == 200
    return res.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# Auth tests

class TestAuth:

    def test_login_success_doctor(self):
        res = client.post(
            "/auth/login",
            data={"username": "doctor", "password": "password123"},
        )
        assert res.status_code == 200
        body = res.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["username"] == "doctor"

    def test_login_success_admin(self):
        res = client.post(
            "/auth/login",
            data={"username": "admin", "password": "adminpass"},
        )
        assert res.status_code == 200

    def test_login_wrong_password(self):
        res = client.post(
            "/auth/login",
            data={"username": "doctor", "password": "wrong"},
        )
        assert res.status_code == 401

    def test_login_unknown_user(self):
        res = client.post(
            "/auth/login",
            data={"username": "ghost", "password": "anything"},
        )
        assert res.status_code == 401

    def test_token_has_expiry_field(self):
        res = client.post(
            "/auth/login",
            data={"username": "doctor", "password": "password123"},
        )
        body = res.json()
        assert "expires_in" in body
        assert body["expires_in"] == 1800


# Health tests

class TestHealth:

    def test_root_returns_online(self):
        res = client.get("/")
        assert res.status_code == 200
        assert res.json()["status"] == "online"

    def test_health_quick_returns_ok(self):
        res = client.get("/health/quick")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_full_health_returns_status(self):
        res = client.get("/health")
        body = res.json()
        assert res.status_code == 200
        assert body["status"] in {"healthy", "degraded", "unhealthy"}
        assert "database" in body
        assert "model" in body
        assert "memory" in body

    def test_full_health_has_version(self):
        res = client.get("/health")
        assert res.json()["version"] == "2.0.0"

    def test_full_health_has_uptime(self):
        res = client.get("/health")
        assert isinstance(res.json()["uptime_s"], int)


# Patient tests

class TestPatientProfile:

    def setup_method(self):
        self.token = get_token()

    def test_list_patients(self):
        res = client.get("/patient/list", headers=auth(self.token))
        assert res.status_code == 200
        assert res.json()["total"] >= 10

    def test_get_patient_profile(self):
        res = client.get("/patient/1/profile", headers=auth(self.token))
        body = res.json()
        assert res.status_code == 200
        assert body["patient_id"] == 1
        assert body["name"] == "James Hooper"
        assert len(body["vital_history"]) == 12

    def test_get_nonexistent_patient(self):
        res = client.get("/patient/9999/profile", headers=auth(self.token))
        assert res.status_code == 404

    def test_vital_keys_present(self):
        res = client.get("/patient/1/profile", headers=auth(self.token))
        vit = res.json()["vital_history"][0]

        for key in ["timestep", "blood_pressure", "glucose", "heart_rate", "inflammation"]:
            assert key in vit

    def test_get_available_treatments(self):
        res = client.get("/patient/treatments", headers=auth(self.token))
        assert res.status_code == 200
        body = res.json()
        assert "medication_A" in body["available_treatments"]
        assert "no_treatment" in body["available_treatments"]


# Prognosis tests

class TestPrognosis:

    def setup_method(self):
        self.token = get_token()

    def test_prognosis_returns_score(self):
        res = client.post(
            "/patient/prognosis",
            json={"vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        body = res.json()
        assert res.status_code == 200
        assert 0.0 <= body["severity_score"] <= 1.0

    def test_prognosis_returns_risk(self):
        res = client.post(
            "/patient/prognosis",
            json={"vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        assert res.json()["risk_level"] in {"Low", "Medium", "High"}

    def test_prognosis_with_patient_id(self):
        res = client.post(
            "/patient/prognosis",
            json={"patient_id": 10, "vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        assert res.json()["saved_to_db"] is True

    def test_prognosis_requires_min_data(self):
        res = client.post(
            "/patient/prognosis",
            json={"vitals_history": SAMPLE_VITALS[:2]},
            headers=auth(self.token),
        )
        assert res.status_code == 422


# Simulation tests

class TestSimulation:

    def setup_method(self):
        self.token = get_token()

    def test_simulate_medication_a(self):
        res = client.post(
            "/patient/simulate-treatment",
            json={"vitals_history": SAMPLE_VITALS, "treatment": "medication_A"},
            headers=auth(self.token),
        )
        body = res.json()
        assert res.status_code == 200
        assert "projected_severity" in body
        assert "improvement" in body
        assert len(body["projected_vitals"]) == 6

    def test_simulate_no_treatment(self):
        res = client.post(
            "/patient/simulate-treatment",
            json={"vitals_history": SAMPLE_VITALS, "treatment": "no_treatment"},
            headers=auth(self.token),
        )
        assert res.status_code == 200

    def test_invalid_treatment(self):
        res = client.post(
            "/patient/simulate-treatment",
            json={"vitals_history": SAMPLE_VITALS, "treatment": "invalid_drug"},
            headers=auth(self.token),
        )
        assert res.status_code == 400

    def test_lifestyle_changes(self):
        res = client.post(
            "/patient/simulate-treatment",
            json={"vitals_history": SAMPLE_VITALS, "treatment": "lifestyle_changes"},
            headers=auth(self.token),
        )
        body = res.json()
        assert body["display_name"] == "Diet & Exercise"
        assert body["recommendation"] in {"Recommended", "Not Recommended"}


# Comparison tests

class TestComparison:

    def setup_method(self):
        self.token = get_token()

    def test_ranked_returns_all(self):
        res = client.post(
            "/compare/ranked",
            json={"vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        body = res.json()
        assert res.status_code == 200
        assert body["total_treatments"] == 4
        assert len(body["ranked_treatments"]) == 4

    def test_ranked_sorted(self):
        res = client.post(
            "/compare/ranked",
            json={"vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        ranked = res.json()["ranked_treatments"]
        scores = [r["final_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_quick_verdict(self):
        res = client.post(
            "/compare/quick-verdict",
            json={"vitals_history": SAMPLE_VITALS},
            headers=auth(self.token),
        )
        body = res.json()
        assert res.status_code == 200
        assert "recommended_treatment" in body
        assert "summary" in body

    def test_history_endpoint(self):
        res = client.get("/compare/history", headers=auth(self.token))
        body = res.json()
        assert res.status_code == 200
        assert "total" in body
        assert "simulations" in body