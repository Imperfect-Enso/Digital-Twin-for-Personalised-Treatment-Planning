from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel


# JWT config

SECRET_KEY = "digital-twin-nextgenhack-secret-2026"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30


# Demo users

DEMO_USERS = {
    "doctor": "password123",
    "admin": "adminpass"
}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    username: str


# Auth scheme (used by FastAPI + Swagger)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# Create JWT

def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": username,
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# Verify JWT

def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return username

    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")


# Login

def login_user(form: OAuth2PasswordRequestForm) -> TokenResponse:
    if form.username not in DEMO_USERS:
        raise HTTPException(status_code=401, detail="Username not found")

    if DEMO_USERS[form.username] != form.password:
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_token(form.username)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=TOKEN_EXPIRE_MINUTES * 60,
        username=form.username
    )