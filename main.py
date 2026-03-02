from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

from api.simulation_routes import router as sim_router
from api.comparison_routes  import router as cmp_router
from middleware.auth import login_user, TokenResponse
from db.database import init_db


app = FastAPI(
    title="Digital Twin — Personalized Treatment API",
    description="""
## Digital Twin ML Backend

A system that creates a digital copy of a patient and simulates
how different treatments will affect their health over time.

---

### How to Use
1. Call **POST /auth/login** to get your access token
2. Click the **Authorize** 🔒 button at the top right of this page
3. Paste your token and click Authorize
4. You're in — all endpoints are now unlocked

### Demo Credentials
- Username: `doctor` · Password: `password123`
- Username: `admin`  · Password: `adminpass`

---

### What Each Group Does
- **/patient** — Feed in vitals, get a severity score or simulate a treatment
- **/compare** — Run all treatments and get them ranked best to worst
- **/auth**    — Login and token management
    """,
    version="1.0.0",
    contact={"name": "Team NextGenHack"}
)


@app.on_event("startup")
def startup():
    # Creates digital_twin.db and seeds patient data on first run
    init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sim_router)
app.include_router(cmp_router)


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Login and get your access token",
    description=(
        "Send your username and password to receive a JWT token. "
        "The token is valid for 30 minutes. After it expires, "
        "just login again to get a fresh one."
    )
)
def login(form: OAuth2PasswordRequestForm = Depends()):
    return login_user(form)


@app.get(
    "/",
    tags=["Health Check"],
    summary="Check if the API is running"
)
def root():
    return {
        "status":  "online",
        "message": "Digital Twin API is up and running!",
        "docs":    "Visit /docs to explore and test all endpoints",
        "login":   "POST /auth/login with your credentials to get started"
    }
