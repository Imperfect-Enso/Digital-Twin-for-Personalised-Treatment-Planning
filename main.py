from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from api.simulation_routes import router as sim_router
from api.comparison_routes  import router as cmp_router
from api.health             import router as health_router       # NEW
from middleware.auth        import login_user, TokenResponse
from middleware.logging     import RequestLoggingMiddleware, log  # NEW
from db.database            import init_db


app = FastAPI(
    title="Digital Twin — Personalized Treatment API",
    version="2.0.0",
    contact={"name": "Team Null2Bit"}
)


# ── Register Swagger lock icon ────────────────────────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Digital Twin — Personalized Treatment API",
        version="2.0.0",
        description="""
## Digital Twin ML Backend

A system that creates a digital copy of a patient and simulates
how different treatments will affect their health over time.

---

### How to Use
1. Call **POST /auth/login** to get your access token
2. Click the **Authorize** 🔒 button at the top right of this page
3. Enter your username and password and click Authorize
4. You're in — all endpoints are now unlocked

### Demo Credentials
- Username: `doctor` · Password: `password123`
- Username: `admin`  · Password: `adminpass`

---

### What Each Group Does
- **/patient** — Feed in vitals, get a severity score or simulate a treatment
- **/compare** — Run all treatments and get them ranked best to worst
- **/health**  — Liveness probe and full system status
- **/auth**    — Login and token management
        """,
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2PasswordBearer": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "/auth/login",
                    "scopes":   {}
                }
            }
        }
    }

    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"OAuth2PasswordBearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    log.info("Digital Twin API started — visit /docs to explore endpoints")


# ── Middleware (order matters: outermost runs first on request) ────────────────
app.add_middleware(RequestLoggingMiddleware)   # NEW — logs every request

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)   # NEW — /health and /health/quick
app.include_router(sim_router)
app.include_router(cmp_router)


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Login and get your access token",
    description=(
        "Send your username and password to receive a JWT token. "
        "The token is valid for 30 minutes. "
        "After it expires, login again to get a fresh one."
    )
)
def login(form: OAuth2PasswordRequestForm = Depends()):
    return login_user(form)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Health Check"],
    summary="Check if the API is running"
)
def root():
    return {
        "status":  "online",
        "version": "2.0.0",
        "message": "Digital Twin API is up and running!",
        "docs":    "Visit /docs to explore and test all endpoints",
        "health":  "GET /health for system status",
        "login":   "POST /auth/login with your credentials to get started"
    }
