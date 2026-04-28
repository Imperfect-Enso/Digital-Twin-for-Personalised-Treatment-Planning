from locust import HttpUser, task, between
import json

# Sample vitals — use one of your preset patients
SAMPLE_VITALS = [
    {"blood_pressure": 121.6, "glucose": 191.5, "heart_rate": 64.1, "inflammation": 1.299},
    {"blood_pressure": 119.4, "glucose": 187.4, "heart_rate": 79.7, "inflammation": 1.756},
    {"blood_pressure": 125.3, "glucose": 183.5, "heart_rate": 68.2, "inflammation": 1.361},
    {"blood_pressure": 119.9, "glucose": 183.2, "heart_rate": 72.1, "inflammation": 1.581},
    {"blood_pressure": 125.1, "glucose": 179.0, "heart_rate": 84.2, "inflammation": 1.183},
    {"blood_pressure": 119.2, "glucose": 173.7, "heart_rate": 85.2, "inflammation": 1.313},
    {"blood_pressure": 123.3, "glucose": 172.1, "heart_rate": 76.7, "inflammation": 1.163},
    {"blood_pressure": 123.3, "glucose": 172.5, "heart_rate": 63.7, "inflammation": 1.178},
    {"blood_pressure": 113.5, "glucose": 163.2, "heart_rate": 65.3, "inflammation": 1.159},
    {"blood_pressure": 125.8, "glucose": 167.3, "heart_rate": 90.8, "inflammation": 1.264},
    {"blood_pressure": 115.7, "glucose": 157.3, "heart_rate": 73.7, "inflammation": 1.224},
    {"blood_pressure": 120.5, "glucose": 158.4, "heart_rate": 88.4, "inflammation": 1.359},
]


import time

class DigitalTwinUser(HttpUser):
    wait_time  = between(1, 3)
    token      = None
    login_time = 0          # ← ADD this attribute

    def on_start(self):
        response = self.client.post(
            "/auth/login",
            data={"username": "doctor", "password": "password123"}
        )
        if response.status_code == 200:
            self.token      = response.json()["access_token"]
            self.login_time = time.time()   # ← record when we logged in

    def auth_headers(self):
        # Token expires in 30 min; re-login after 25 min to be safe
        if time.time() - self.login_time > 1500:
            self.on_start()
        return {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def get_patient_profile(self):
        """Most common action — doctors checking patient profiles"""
        import random
        patient_id = random.randint(1, 10)
        self.client.get(
            f"/patient/{patient_id}/profile",
            headers=self.auth_headers()
        )

    @task(2)
    def compare_treatments(self):
        """Second most common — running treatment comparisons"""
        self.client.post(
            "/compare/ranked",
            json={"vitals_history": SAMPLE_VITALS},
            headers=self.auth_headers()
        )

    @task(1)
    def simulate_treatment(self):
        """Least frequent — full simulation is heavier"""
        self.client.post(
            "/patient/simulate-treatment",
            json={
                "vitals_history": SAMPLE_VITALS,
                "treatment": "medication_A"
            },
            headers=self.auth_headers()
        )

import psutil
import threading
import time

def monitor_cpu():
    """Prints CPU usage every 5 seconds while Locust is running"""
    while True:
        cpu    = psutil.cpu_percent(interval=1)
        mem    = psutil.virtual_memory().percent
        cores  = psutil.cpu_count()
        print(f"\n[SYSTEM MONITOR] CPU: {cpu}% | RAM: {mem}% | Cores: {cores}")
        time.sleep(5)

# Start monitor in background thread when file loads
monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
monitor_thread.start()