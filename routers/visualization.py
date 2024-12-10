
# backend/routers/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.base import get_db
from database.models import Dataset

router = APIRouter(prefix="/visualize", tags=["visualization"])

@router.get("/scatter")
def create_scatter_plot(
    dataset_id: int, 
    x_column: str, 
    y_column: str,
    db: Session = Depends(get_db)
):
    """
    Generate a scatter plot for two numerical columns
    """
    # Fetch dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read dataset
    try:
        if dataset.file_path.endswith('.csv'):
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_json(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
    
    # Validate columns
    if x_column not in df.columns or y_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column names")
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.6)
    plt.title(f"Scatter Plot: {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "plot": plot_data,
        "message": "Scatter plot generated successfully"
    }

@router.get("/histogram")
def create_histogram(
    dataset_id: int, 
    column: str,
    bins: int = 10,
    db: Session = Depends(get_db)
):
    """
    Generate a histogram for a numerical column
    """
    # Fetch dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read dataset
    try:
        if dataset.file_path.endswith('.csv'):
            df = pd.read_csv(dataset.file_path)
        else:
            df = pd.read_json(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
    
    # Validate column
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column name")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "plot": plot_data,
        "message": "Histogram generated successfully"
    }