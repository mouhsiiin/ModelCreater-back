from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from routers import dataset, ml_algorithm, user, visualization, project, preprocessing
from database.base import engine, get_db, Base
from datetime import timedelta
from security.auth import create_access_token, authenticate_user, check_admin
from models.users import Token
from passlib.context import CryptContext

# Initialize FastAPI app
app = FastAPI(title="Machine Learning Web Application")

# Create database tables
Base.metadata.create_all(bind=engine)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



# Routers
app.include_router(dataset.router)
app.include_router(ml_algorithm.router)
app.include_router(visualization.router)
app.include_router(user.router)
app.include_router(project.router)
app.include_router(preprocessing.router)



# create admin user
db = next(get_db())
check_admin(db)

@app.get("/")
def read_root():
    return {"message": "Welcome to ML Web Application Backend"}




@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={"username": user.username, "email": user.email, "is_active": user.disabled, "id": user.id},
    )
    return {"access_token": access_token, "token_type": "bearer"}