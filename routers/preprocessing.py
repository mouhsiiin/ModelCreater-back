from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.preprocessing import PreprocessingConfiguration
from services.data_preprocessing import PreprocessingService
from typing import List, Dict, Any


router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])





@router.post("/configure")
async def configure_preprocessing(
    config: PreprocessingConfiguration,
    db: Session = Depends(get_db)
):
    try:
        preprocessing_service = PreprocessingService(db)
        config_id = preprocessing_service.store_configuration(config)
        return {
            "status": "success",
            "message": "Preprocessing configured",
            "config_id": config_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execute")
async def execute_preprocessing(
    data: List[Dict[str, Any]],
    config_id: str,
    db: Session = Depends(get_db)
):
    try:
        preprocessing_service = PreprocessingService(db)
        config = preprocessing_service.get_configuration(config_id)
        
        result_df = preprocessing_service.execute_preprocessing(
            data,
            config.options
        )
        
        return {
            "status": "success",
            "message": "Preprocessing completed",
            "data": result_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))