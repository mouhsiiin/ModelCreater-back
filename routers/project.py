from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database.base import get_db
from models.projects import Project, ProjectCreate, ProjectUpdate, ProjectDB
from models.users import User
from security.auth import get_current_user

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


# get recent projects
@router.get("/recent", response_model=List[Project])
def get_recent_projects(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the most recent projects"""
    if current_user:
        projects = db.query(ProjectDB).filter(ProjectDB.owner_id == current_user.id).order_by(ProjectDB.created_at.desc()).limit(limit).all()
    else:
        # for guest users, return public projects
        projects = db.query(ProjectDB).filter(ProjectDB.is_public == True).order_by(ProjectDB.created_at.desc()).limit(limit).all()
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