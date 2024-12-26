from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.preprocessing import PreprocessingConfigurationCreate
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from models.projects import ProjectDB
from services.data_preprocessing import PreprocessingService
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import os

UPLOAD_DIRECTORY = "uploads"
router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])

@router.post("/process/{project_id}")
async def process_data(
    config: PreprocessingConfigurationCreate,
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Endpoint to store preprocessing configuration and execute preprocessing from uploaded file.
    
    args:
        config (PreprocessingConfigurationCreate): Configuration for preprocessing
        project_id (int): Project ID to get dataset from
        db (Session): Database session
    """
    
    # Get dataset ID from project
    dataset_id = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).first().id
    
    if not dataset_id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    print(config.model_dump())
    
    try:
        preprocessing_service = PreprocessingService(db)
        config_id = preprocessing_service.store_configuration(config.model_dump())
        
        # Read dataset file
        file_path = os.path.join(UPLOAD_DIRECTORY, f"dataset_{dataset_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        data = pd.read_csv(file_path)
        
        preprocessed_dataset = PreprocessedDatasetDB(
            dataset_id=dataset_id,
            config_id=config_id,
            status="pending",
            location="",
            metadata={
                "started_at": datetime.utcnow().isoformat(),
                "original_rows": len(data)
            }
        )
        db.add(preprocessed_dataset)
        db.flush()

        try:
            print("Processing data...")
            print("Config: ", config)
            result_df = preprocessing_service.execute_preprocessing(data.to_dict('records'), config.model_dump()["options"])
            
            storage_location = f"preprocessed/dataset_{dataset_id}/version_{preprocessed_dataset.id}"
            preprocessing_service.save_preprocessed_data(result_df, storage_location)

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
                "config_id": config_id,
                "preprocessed_dataset_id": preprocessed_dataset.id,
                "location": storage_location,
                "metadata": preprocessed_dataset.metadata
            }

        except Exception as process_error:
            preprocessed_dataset.status = "failed"
            preprocessed_dataset.metadata.update({
                "error": str(process_error),
                "failed_at": datetime.utcnow().isoformat()
            })
            db.commit()
            raise process_error

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")