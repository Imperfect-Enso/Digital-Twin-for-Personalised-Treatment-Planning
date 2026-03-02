from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel


# JWT configuration
SECRET_KEY           = "digital-twin-nextgenhack-secret-2026"
ALGORITHM            = "HS256"
TOKEN_EXPIRE_MINUTES = 30

# Authorised demo accounts — { username: password }
DEMO_USERS = {
    "doctor": "password123",
    "admin":  "adminpass"
}


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    expires_in:   int
    username:     str


# Points FastAPI and Swagger UI to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_token(username: str) -> str:
    """Signs and returns a JWT token that expires in TOKEN_EXPIRE_MINUTES."""
    expire  = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    """
    FastAPI dependency — validates the Bearer token on every protected request.
    Returns the username on success, raises HTTP 401 on failure.
    """
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token — no username found")

        return username

    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid — please login again")


def login_user(form: OAuth2PasswordRequestForm) -> TokenResponse:
    """Validates credentials and returns a signed token."""
    if form.username not in DEMO_USERS:
        raise HTTPException(status_code=401, detail="Username not found")

    if DEMO_USERS[form.username] != form.password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_token(form.username)

    return TokenResponse(
        access_token = token,
        token_type   = "bearer",
        expires_in   = TOKEN_EXPIRE_MINUTES * 60,
        username     = form.username
    )
