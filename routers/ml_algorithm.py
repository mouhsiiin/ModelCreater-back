# routers/ml_algorithms.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from typing import Optional, Dict, Any

from database.base import get_db
from models.ml_models import MLModelDB
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from services.ml_service import MLTrainingService, MLPredictionService

router = APIRouter(prefix="/ml", tags=["machine_learning"])
ml_service = MLTrainingService()

@router.post("/train")
def train_model(
    project_id: int,
    algorithm: str,
    target_column: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    scale_features: bool = True,
    db: Session = Depends(get_db)
):
    """Train a machine learning model on a dataset"""
    # Fetch dataset
    raw_dataset = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).first()
    
    # Fetch the latest preprocessed dataset
    dataset = db.query(PreprocessedDatasetDB).filter(PreprocessedDatasetDB.dataset_id == raw_dataset.id).order_by(PreprocessedDatasetDB.id.desc()).first()
    
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read dataset
    try:
        if dataset.location.endswith('.csv'):
            df = pd.read_csv(dataset.location)
        else:
            df = pd.read_json(dataset.location)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error reading dataset: {str(e)}"
        )
        
    
    # Validate target column
    if not target_column or target_column not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail="Invalid or missing target column"
        )
    
    try:
        # Train model
        result = ml_service.train(
            df=df,
            algorithm=algorithm,
            target_column=target_column,
            dataset_id=dataset.id,
            hyperparameters=hyperparameters,
            test_size=test_size,
            scale_features=scale_features
        )
        
        
        # Save to database
        ml_model = MLModelDB(
            dataset_id=dataset.id,
            algorithm_name=algorithm,
            model_path=result["model_path"],
            performance_metrics=result["metrics"],
            target_column=target_column,
            feature_names=result["feature_names"]
        )
        db.add(ml_model)
        db.commit()
        db.refresh(ml_model)
        
        return {
            "message": "Model trained successfully",
            "model_id": ml_model.id,
            "algorithm": algorithm,
            "metrics": result["metrics"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )


@router.post("/predict/{model_id}")
def predict(
    model_id: int,
    input_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Make predictions using a trained model"""
    # Initialize prediction service
    prediction_service = MLPredictionService()
    
    # Fetch model from database
    ml_model = db.query(MLModelDB).filter(MLModelDB.id == model_id).first()
    if not ml_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        predictions = prediction_service.predict(ml_model.model_path, input_df)
        
        return {
            "predictions": predictions.tolist(),
            "model_id": model_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )