# 🧬 Digital Twin — Personalized Treatment Planning

> NextGenHack 2026 | Problem Statement 5

A LSTM-based time series model that creates a virtual copy of a patient
from their 12-month vital history, simulates future health trajectories
under different treatment plans, and recommends the optimal treatment
using a weighted clinical scoring engine.

---

## 🏗️ System Architecture
```
Patient Vitals (12 months)
        ↓
LSTM Neural Network (TensorFlow)
        ↓
Treatment Simulation Engine
        ↓
3-Factor Weighted Scoring
(Severity 50% + Vitals 30% + Speed 20%)
        ↓
Ranked Treatment Recommendations
        ↓
REST API (FastAPI) + Dashboard (HTML/JS)
```

---

## 🚀 Features

- **LSTM Model** — trained on longitudinal patient vital sequences
- **Treatment Simulator** — projects 4 treatments 6 months forward
- **Scoring Engine** — multi-factor clinical ranking system
- **REST API** — 9 endpoints with JWT authentication and Swagger docs
- **SQLite Database** — persists all simulations with timestamps
- **Interactive Dashboard** — 5-page frontend with live Chart.js graphs

---

## 👥 Team

| Member | Role |
|--------|------|
| Member 1 | Patient Database & Data Pipeline |
| Member 2 | ML Model, Treatment Simulator, API |
| Member 3 | Scoring Engine, Auth, Frontend |

---

## 🛠️ Tech Stack

**Backend:** Python, FastAPI, TensorFlow, SQLAlchemy, SQLite
**ML:** LSTM (Keras), NumPy, Pandas, Scikit-learn
**Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js
**Auth:** JWT (python-jose), OAuth2

---

## ⚙️ How to Run

### Backend
```bash
cd digital_twin_ml
pip install -r requirements.txt
python ml/train.py
uvicorn main:app --reload
```

API runs at: http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs

### Frontend
Open `digital_twin_frontend/index.html` in your browser.

Demo credentials:
- Username: `doctor`
- Password: `password123`

---

## 📁 Project Structure
```
digital_twin_ml/          ← Backend
├── data/                 ← Data pipeline
├── ml/                   ← LSTM model + simulator
├── core/                 ← Scoring engine
├── api/                  ← FastAPI routes
├── middleware/           ← JWT auth
├── db/                   ← SQLite database
└── main.py               ← App entry point

digital_twin_frontend/    ← Frontend
├── index.html            ← Login
├── dashboard.html        ← Doctor Dashboard
├── patient.html          ← Patient Overview
├── simulation.html       ← Treatment Simulation
└── comparison.html       ← Visual Comparison
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/login | Get JWT token |
| GET | /patient/list | All 10 patients |
| GET | /patient/{id}/profile | Patient + vitals |
| POST | /patient/prognosis | LSTM severity score |
| POST | /compare/ranked | Ranked treatments |
| POST | /compare/quick-verdict | Best treatment |
| GET | /compare/history | Past simulations |