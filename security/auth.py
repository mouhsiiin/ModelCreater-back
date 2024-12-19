from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from models.users import User, UserDB
from passlib.context import CryptContext
from database.base import get_db
from utils.helpers import get_password_hash as hash_password



# Secret key and settings
SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user_by_username_or_email(db: Session, identifier: str) -> Optional[UserDB]:
    return db.query(UserDB).filter(
        (UserDB.username == identifier) | (UserDB.email == identifier)
    ).first()

# Authenticate user
def authenticate_user(db: Session, identifier: str, password: str):
    user = get_user_by_username_or_email(db, identifier)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# Create access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt



def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
    
    user = db.query(UserDB).filter(UserDB.username == username).first()  # Changed User to UserDB
    if user is None:
        raise credentials_exception
    return user


# Decode access token for additional data extraction
def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        email: str = payload.get("email")
        if username is None or email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "email": email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
# check if admin exists
def check_admin(db: Session = Depends(get_db)):
    admin = db.query(UserDB).filter(UserDB.username == "admin").first()
    if not admin:
        admin = UserDB(
            username="admin",
            email="admin@email.com",
            hashed_password=hash_password("admin"),
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
    return admin
