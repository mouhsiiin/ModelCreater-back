from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database.base import get_db
from models.projects import Project, ProjectCreate, ProjectUpdate, ProjectDB
from models.preprocessed_datasets import PreprocessedDatasetDB
from models.datasets import DatasetDB
from models.users import User
from security.auth import get_current_user
import pandas as pd

router = APIRouter(
    prefix="/projects",
    tags=["projects"]
)

@router.post("/", response_model=Project)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new project"""
    db_project = project.create_db_instance(owner_id=current_user.id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.get("/", response_model=List[Project])
def get_projects(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all projects with pagination"""
    projects = db.query(ProjectDB).offset(skip).limit(limit).all()
    return projects


@router.get("/my", response_model=List[Project])
def get_my_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all projects owned by the current user"""
    projects = db.query(ProjectDB).filter(ProjectDB.owner_id == current_user.id).all()
    return projects

# get the colums of the latest preprocessed dataset
@router.get("/columns/{project_id}")
def get_project_columns(
    project_id: int,
    db: Session = Depends(get_db),
):
    """Get the columns of the latest preprocessed dataset"""
    raw_dataset = db.query(DatasetDB).filter(DatasetDB.project_id == project_id).first()
    
    dataset = db.query(PreprocessedDatasetDB).filter(PreprocessedDatasetDB.dataset_id == raw_dataset.id).order_by(PreprocessedDatasetDB.id.desc()).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # read dataset csv
    
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
        
    return df.columns.tolist()



# get recent projects
@router.get("/recent", response_model=List[Project])
def get_recent_projects(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the most recent projects"""
    projects = db.query(ProjectDB).filter(ProjectDB.owner_id == current_user.id).order_by(ProjectDB.created_at.desc()).limit(limit).all()
    return projects

@router.get("/{project_id}", response_model=Project)
def get_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific project by ID"""
    project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.put("/{project_id}", response_model=Project)
def update_project(
    project_id: int,
    project_update: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a project"""
    db_project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    updated_project = project_update.update_db_instance(db_project)
    db.add(updated_project)
    db.commit()
    db.refresh(updated_project)
    return updated_project

@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a project"""
    db_project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(db_project)
    db.commit()
    return {"message": "Project deleted successfully"}