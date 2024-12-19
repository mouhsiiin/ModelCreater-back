
# backend/routers/ml_algorithms.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import os
import json

from database.base import get_db
from models.ml_models import MLModelDB
from models.datasets import DatasetDB

router = APIRouter(prefix="/ml", tags=["machine_learning"])

MODELS_DIRECTORY = "trained_models/"
os.makedirs(MODELS_DIRECTORY, exist_ok=True)

@router.post("/train")
def train_model(
    dataset_id: int, 
    algorithm: str = "linear_regression", 
    target_column: str = None,
    test_size: float = 0.2,
    db: Session = Depends(get_db)
):
    """
    Train a machine learning model on a dataset
    """
    # Fetch dataset
    dataset = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read dataset
    try:
        if dataset.file_path.endswith('.csv'):
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_json(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
    
    # Validate target column
    if not target_column or target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid or missing target column")
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model based on algorithm
    if algorithm == "linear_regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        }
    elif algorithm == "logistic_regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported algorithm")
    
    # Save model
    model_filename = f"{algorithm}_{dataset_id}_{target_column}.pkl"
    model_path = os.path.join(MODELS_DIRECTORY, model_filename)
    joblib.dump(model, model_path)
    
    # Save to database
    ml_model = MLModelDB(
        dataset_id=dataset_id,
        algorithm_name=algorithm,
        model_path=model_path,
        performance_metrics=metrics
    )
    db.add(ml_model)
    db.commit()
    db.refresh(ml_model)
    
    return {
        "message": "Model trained successfully",
        "model_id": ml_model.id,
        "algorithm": algorithm,
        "metrics": metrics
    }
