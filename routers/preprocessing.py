from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session 
from database.base import get_db
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from services.data_preprocessing import PreprocessingService, PreprocessingError
from typing import Dict, Any
from datetime import datetime, timezone
import pandas as pd
import os
import json

UPLOAD_DIRECTORY = "uploads"
router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])

@router.get("/options")
async def get_preprocessing_options(db: Session = Depends(get_db)):
    """Get available preprocessing options and their valid values"""
    preprocessing_service = PreprocessingService(db)
    return preprocessing_service.get_available_options()

@router.post("/process/{project_id}")
async def process_data(
    options: dict,
    preview_stats: dict,
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Endpoint to store preprocessing configuration and execute preprocessing from uploaded file.
    
    Args:
        options: Configuration for preprocessing
        preview_stats: Preview statistics for preprocessing
        project_id (int): Project ID to get dataset from
        db (Session): Database session
    """
    try:
        # Get the latest dataset ID from project 
        dataset = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).order_by(DatasetDB.id.desc()).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        preprocessing_service = PreprocessingService(db)
        
        # Validate options before proceeding
        try:
            preprocessing_service.validate_options(options)
        except PreprocessingError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Store configuration
        config_id = preprocessing_service.store_configuration(options, preview_stats)
        
        # Read dataset file
        file_path = dataset.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        data = pd.read_csv(file_path)
        
        # Create preprocessed dataset record
        preprocessed_dataset = PreprocessedDatasetDB(
            dataset_id=dataset.id,
            config_id=config_id,
            status="pending",
            location="",
            metadata={
                "started_at": datetime.now(timezone.utc).isoformat(),
                "original_rows": len(data),
                "original_columns": list(data.columns)
            }
        )
        db.add(preprocessed_dataset)
        db.flush()

        try:
            # Execute preprocessing
            result, summary = preprocessing_service.execute_preprocessing(
                data.to_dict('records'), 
                options
            )
            
            # Handle split datasets if returned
            if isinstance(result, tuple):
                train_df, test_df = result
                if len(train_df) == 0 or len(test_df) == 0:
                    raise ValueError("No data left after preprocessing splits")
                    
                base_dir = os.path.join(UPLOAD_DIRECTORY, f"project_{project_id}")
                os.makedirs(base_dir, exist_ok=True)
                
                # Save train dataset
                train_location = os.path.join(base_dir, f"version_{preprocessed_dataset.id}_train.csv")
                train_df.to_csv(train_location, index=False)
                
                # Save test dataset
                test_location = os.path.join(base_dir, f"version_{preprocessed_dataset.id}_test.csv")
                test_df.to_csv(test_location, index=False)
                
                storage_location = {
                    "train": train_location,
                    "test": test_location
                }
                result_df = train_df  # Use train data for metadata
                
            else:
                if len(result) == 0:
                    raise ValueError("No data left after preprocessing")
                    
                base_dir = os.path.join(UPLOAD_DIRECTORY, f"project_{project_id}")
                os.makedirs(base_dir, exist_ok=True)
                
                storage_location = os.path.join(base_dir, f"version_{preprocessed_dataset.id}.csv")
                result.to_csv(storage_location, index=False)
                result_df = result

            # Update preprocessed dataset record
            preprocessed_dataset.status = "completed"
            preprocessed_dataset.location = json.dumps(storage_location) if isinstance(storage_location, dict) else storage_location
            
            # Update metadata with summary and additional info
            preprocessed_dataset.metadata.update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "processed_rows": len(result_df),
                "processed_columns": list(result_df.columns),
                "preprocessing_summary": summary
            })
            
            db.commit()

            return {
                "status": "success",
                "config_id": config_id,
                "preprocessed_dataset_id": preprocessed_dataset.id,
                "location": storage_location,
                "metadata": preprocessed_dataset.metadata
            }

        except Exception as process_error:
            preprocessed_dataset.status = "failed"
            preprocessed_dataset.metadata.update({
                "error": str(process_error),
                "failed_at": datetime.now(timezone.utc).isoformat()
            })
            db.commit()
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(process_error)}")

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")