from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.preprocessing import PreprocessingConfiguration
from schemas.preprocessing import PreprocessingConfigurationCreate, PreprocessingExecutionResult
from services.data_preprocessing import PreprocessingService
from typing import List, Dict, Any

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


@router.post("/execute", response_model=PreprocessingExecutionResult)
async def execute_preprocessing(
    data: List[Dict[str, Any]],
    config_id: str,
    db: Session = Depends(get_db)
):
    """
    Endpoint to execute preprocessing on provided data using a specific configuration.

    Args:
        data: List[Dict[str, Any]] - The raw data to preprocess.
        config_id: str - ID of the preprocessing configuration to use.
        db: Database session dependency.

    Returns:
        JSON response with status, message, and preprocessed data.
    """
    try:
        preprocessing_service = PreprocessingService(db)
        config = preprocessing_service.get_configuration(config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")

        result_df = preprocessing_service.execute_preprocessing(data, config.options)

        return {
            "status": "success",
            "message": "Preprocessing completed successfully.",
            "data": result_df.to_dict(orient="records"),
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to execute preprocessing: {str(e)}")
