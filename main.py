from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import dataset, ml_algorithms, visualization
from database.base import engine
from database import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Machine Learning Web Application")

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dataset.router)
app.include_router(ml_algorithms.router)
app.include_router(visualization.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to ML Web Application Backend"}