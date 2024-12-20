import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from database.base import get_db
from models.datasets import DatasetDB

router = APIRouter(prefix="/visualize", tags=["visualization"])

def load_dataset(dataset_id: int, db: Session) -> pd.DataFrame:
    """Helper function to load and validate dataset"""
    dataset = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        if dataset.file_path.endswith('.csv'):
            return pd.read_csv(dataset.file_path)
        else:
            return pd.read_json(dataset.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

def create_plot_response(plt) -> dict:
    """Helper function to convert plot to base64"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return {"plot": plot_data, "message": "Plot generated successfully"}

@router.get("/scatter")
def create_scatter_plot(
    dataset_id: int, 
    x_column: str, 
    y_column: str,
    hue_column: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Generate a scatter plot with optional color grouping"""
    df = load_dataset(dataset_id, db)
    
    if x_column not in df.columns or y_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column names")
    
    plt.figure(figsize=(10, 6))
    if hue_column:
        if hue_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid hue column name")
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
    else:
        sns.scatterplot(data=df, x=x_column, y=y_column)
    
    plt.title(f"Scatter Plot: {x_column} vs {y_column}")
    return create_plot_response(plt)

@router.get("/histogram")
def create_histogram(
    dataset_id: int, 
    column: str,
    bins: int = 10,
    kde: bool = False,
    db: Session = Depends(get_db)
):
    """Generate a histogram with optional kernel density estimation"""
    df = load_dataset(dataset_id, db)
    
    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column name")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=kde)
    plt.title(f"Histogram of {column}")
    return create_plot_response(plt)

@router.get("/boxplot")
def create_boxplot(
    dataset_id: int,
    y_column: str,
    x_column: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Generate a box plot with optional grouping"""
    df = load_dataset(dataset_id, db)
    
    if y_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column name")
    
    plt.figure(figsize=(10, 6))
    if x_column:
        if x_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid grouping column")
        sns.boxplot(data=df, x=x_column, y=y_column)
    else:
        sns.boxplot(data=df, y=y_column)
    
    plt.title(f"Box Plot of {y_column}")
    return create_plot_response(plt)

@router.get("/lineplot")
def create_lineplot(
    dataset_id: int,
    x_column: str,
    y_columns: List[str],
    db: Session = Depends(get_db)
):
    """Generate a line plot for multiple y columns against an x column"""
    df = load_dataset(dataset_id, db)
    
    if x_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid x column name")
    
    for col in y_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Invalid y column name: {col}")
    
    plt.figure(figsize=(12, 6))
    for col in y_columns:
        plt.plot(df[x_column], df[col], label=col)
    
    plt.title(f"Line Plot over {x_column}")
    plt.legend()
    plt.xticks(rotation=45)
    return create_plot_response(plt)

@router.get("/barplot")
def create_barplot(
    dataset_id: int,
    x_column: str,
    y_column: str,
    aggregation: str = "mean",
    db: Session = Depends(get_db)
):
    """Generate a bar plot with aggregated values"""
    df = load_dataset(dataset_id, db)
    
    if x_column not in df.columns or y_column not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid column names")
    
    valid_aggregations = ["mean", "sum", "count", "median"]
    if aggregation not in valid_aggregations:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid aggregation. Must be one of: {', '.join(valid_aggregations)}"
        )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=x_column, y=y_column, estimator=getattr(pd, aggregation))
    plt.title(f"Bar Plot: {aggregation} of {y_column} by {x_column}")
    plt.xticks(rotation=45)
    return create_plot_response(plt)

@router.get("/heatmap")
def create_heatmap(
    dataset_id: int,
    columns: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """Generate a correlation heatmap for numerical columns"""
    df = load_dataset(dataset_id, db)
    
    if columns:
        if not all(col in df.columns for col in columns):
            raise HTTPException(status_code=400, detail="Invalid column names")
        numeric_df = df[columns].select_dtypes(include=['int64', 'float64'])
    else:
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if numeric_df.empty:
        raise HTTPException(status_code=400, detail="No numerical columns available")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    return create_plot_response(plt)