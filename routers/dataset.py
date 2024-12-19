import os
import pandas as pd
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from database.base import get_db
from models.datasets import DatasetDB 

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a dataset file (CSV or JSON)
    """
    # Validate file type
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and JSON supported.")
    
    # Save file
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Read and validate dataset
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Create database entry
    db_dataset = DatasetDB(
        name=file.filename,
        file_path=file_path,
        dataset_metadata={
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": str(df.dtypes)
        }
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    return {
        "message": "Dataset uploaded successfully",
        "dataset_id": db_dataset.id,
        "filename": file.filename,
        "columns": list(df.columns)
    }

@router.get("/")
def list_datasets(db: Session = Depends(get_db)):
    """
    Retrieve list of uploaded datasets
    """
    datasets = db.query(DatasetDB).all()
    return [
        {
            "id": dataset.id,
            "name": dataset.name,
            "created_at": dataset.created_at,
            "metadata": dataset.dataset_metadata
        } for dataset in datasets
    ]

@router.get("/{dataset_id}")
def get_dataset_details(dataset_id: int, db: Session = Depends(get_db)):
    """
    Get details of a specific dataset
    """
    dataset = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read dataset to provide preview
    try:
        if dataset.file_path.endswith('.csv'):
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_json(dataset.file_path)
        
        return {
            "id": dataset.id,
            "name": dataset.name,
            "file_path": dataset.file_path,
            "metadata": dataset.dataset_metadata,
            "sample_data": df.head(10).to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
