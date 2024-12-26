from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.preprocessing import PreprocessingConfigurationCreate
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from models.projects import ProjectDB
from services.data_preprocessing import PreprocessingService
from typing import Dict, Any
from datetime import datetime, timezone
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
    dataset = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).first()
    dataset_id = dataset.id
    
    
    if not dataset_id:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    print("store config")
    
    try:
        preprocessing_service = PreprocessingService(db)
        config_id = preprocessing_service.store_configuration(config.model_dump())
        
        print("config_id: ", config_id)
        
        
        # Read dataset file
        file_path = dataset.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        data = pd.read_csv(file_path)
        
        preprocessed_dataset = PreprocessedDatasetDB(
            dataset_id=dataset_id,
            config_id=config_id,
            status="pending",
            location="",
            metadata={
                "started_at": datetime.now(timezone.utc).isoformat(),
                "original_rows": len(data)
            }
        )
        db.add(preprocessed_dataset)
        db.flush()
        
        print("config is stored")

        try:
            print("Processing data...")
            result_df = preprocessing_service.execute_preprocessing(data.to_dict('records'), config.model_dump()["options"])
            
            if len(result_df) == 0:
                raise ValueError("No data left after preprocessing")
            
            # Ensure the base directory exists
            base_dir = os.path.join(UPLOAD_DIRECTORY, f"project_{project_id}")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)



            # Construct the storage location
            if preprocessed_dataset.id is None:
                raise ValueError("preprocessed_dataset.id is None; cannot construct storage path.")

            storage_location = os.path.join(base_dir, f"version_{preprocessed_dataset.id}.csv")

            # Save the DataFrame
            try:
                result_df.to_csv(storage_location, index=False)
                print(f"Data successfully saved to {storage_location}")
            except Exception as e:
                print(f"Failed to save data: {e}")

            preprocessed_dataset.status = "completed"
            preprocessed_dataset.location = storage_location
            preprocessed_dataset.metadata.update({
                "completed_at": datetime.now(timezone.utc).isoformat(),
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