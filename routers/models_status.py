from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from models.ml_models import MLModelDB
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from models.projects import ProjectDB
from database.base import get_db
from security.auth import get_current_user
from models.users import User




router = APIRouter(prefix="/models", tags=["models"])

@router.get("/latest/project/{project_id}")
def get_models_by_project_id(
    project_id: int,
    db = Depends(get_db)
):
    """Get the latest trained model for a project"""
    dataset = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).first()
    
    print(dataset.id)
    
    if not dataset:
        return {"message": "No dataset found for this project"}
    
    preprocced_dataset = db.query(PreprocessedDatasetDB).filter(PreprocessedDatasetDB.dataset_id == dataset.id).order_by(PreprocessedDatasetDB.id.desc()).first()
    
    if not preprocced_dataset:
        return {"message": "No preprocessed dataset found for this project"}
    
    model = db.query(MLModelDB).filter(MLModelDB.dataset_id == preprocced_dataset.id).order_by(MLModelDB.id.desc()).first()    
    if not model:
        return {"message": "No models found for this project"}
    
    return model


# download model file
@router.get("/download/{model_id}")
def download_model_file(
    model_id: int,
    db = Depends(get_db)
):
    """Download the model file"""
    model = db.query(MLModelDB).filter(MLModelDB.id == model_id).first()
    
    if not model:
        return {"message": "Model not found"}
    
    return FileResponse(model.model_path, media_type='application/octet-stream', filename=model.model_path)


# get models for authed user
@router.get("/my")
def get_models_by_user(
    db = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all models for the current user"""
    user_projects = db.query(ProjectDB).filter(ProjectDB.owner_id == current_user.id).all()
    project_ids = [project.id for project in user_projects]
    
    
    datasets = db.query(DatasetDB).filter(DatasetDB.project_id.in_(project_ids)).all()
    datasets_ids = [dataset.id for dataset in datasets]
    
    print(datasets_ids)
    
    models = db.query(MLModelDB).filter(MLModelDB.dataset_id.in_(datasets_ids)).all()
    
    return models