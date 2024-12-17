from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from routers import dataset, ml_algorithms, visualization, users
from database.base import engine, get_db
from database import models
from datetime import timedelta
from security.auth import create_access_token, get_current_user
from schemas.user import Token
from passlib.context import CryptContext

# Initialize FastAPI app
app = FastAPI(title="Machine Learning Web Application")

# Create database tables
models.Base.metadata.create_all(bind=engine)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token settings
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Routers
app.include_router(dataset.router)
app.include_router(ml_algorithms.router)
app.include_router(visualization.router)
app.include_router(users.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to ML Web Application Backend"}


# Authenticate user from the database
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return None
    return user


# Token endpoint
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
