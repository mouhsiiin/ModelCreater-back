from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.preprocessing import PreprocessingConfiguration, PreprocessingConfigurationCreate
from models.preprocessed_datasets import PreprocessedDataset
from services.data_preprocessing import PreprocessingService
from typing import List, Dict, Any
import os
from datetime import datetime


router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])

@router.post("/configure")
async def configure_preprocessing(
    config: PreprocessingConfigurationCreate,
    db: Session = Depends(get_db)
):
    """
    Endpoint to store preprocessing configuration.
    
    Args:
        config: PreprocessingConfigurationCreate - Configuration details provided by the user.
        db: Database session dependency.
    
    Returns:
        JSON response with status and configuration ID.
    """
    try:
        preprocessing_service = PreprocessingService(db)
        config_id = preprocessing_service.store_configuration(config.dict())
        return {
            "status": "success",
            "message": "Preprocessing configuration stored successfully.",
            "config_id": config_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to store configuration: {str(e)}")

@router.post("/execute")
async def execute_preprocessing(
    data: List[Dict[str, Any]],
    config_id: str,
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """
    Endpoint to execute preprocessing on provided data using a specific configuration.
    
    Args:
        data: List[Dict[str, Any]] - The raw data to preprocess.
        config_id: str - ID of the preprocessing configuration to use.
        dataset_id: int - ID of the dataset being preprocessed.
        db: Database session dependency.
    
    Returns:
        JSON response with status, message, and preprocessed dataset information.
    """
    try:
        preprocessing_service = PreprocessingService(db)
        config = preprocessing_service.get_configuration(config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")

        # Create a new preprocessed dataset entry
        preprocessed_dataset = PreprocessedDataset(
            dataset_id=dataset_id,
            config_id=config_id,
            status="pending",
            location="",  # Will be updated after processing
            metadata={
                "started_at": datetime.utcnow().isoformat(),
                "original_rows": len(data)
            }
        )
        db.add(preprocessed_dataset)
        db.flush()  # Get the ID without committing

        try:
            # Execute preprocessing
            result_df = preprocessing_service.execute_preprocessing(data, config.options)
            
            # Save the preprocessed data to a storage location
            storage_location = f"preprocessed/dataset_{dataset_id}/version_{preprocessed_dataset.id}"
            preprocessing_service.save_preprocessed_data(result_df, storage_location)

            # Update the preprocessed dataset entry
            preprocessed_dataset.status = "completed"
            preprocessed_dataset.location = storage_location
            preprocessed_dataset.metadata.update({
                "completed_at": datetime.utcnow().isoformat(),
                "processed_rows": len(result_df),
                "columns": list(result_df.columns)
            })
            
            db.commit()

            return {
                "status": "success",
                "message": "Preprocessing completed successfully.",
                "preprocessed_dataset_id": preprocessed_dataset.id,
                "location": storage_location,
                "metadata": preprocessed_dataset.metadata
            }

        except Exception as process_error:
            # Update the preprocessed dataset entry with error status
            preprocessed_dataset.status = "failed"
            preprocessed_dataset.metadata.update({
                "error": str(process_error),
                "failed_at": datetime.utcnow().isoformat()
            })
            db.commit()
            raise process_error

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to execute preprocessing: {str(e)}")