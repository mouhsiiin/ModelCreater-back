from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from schemas.user import UserInDB
from passlib.context import CryptContext

# Secret key and settings
SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Dummy user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": "$2b$12$W0H0jg1bOdz5pZJuyRq4HuDZUj6v/2f9cCUECEg.ZRYTa2FZXNRyC",  # bcrypt hash of "password123"
        "email": "testuser@example.com",
        "disabled": False,
    }
}

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Authenticate user
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return UserInDB(**user)


# Create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Get current user dependency
def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return UserInDB(**user)
