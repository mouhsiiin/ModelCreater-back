
from sqlalchemy import Column, Integer, String, JSON, DateTime, Boolean
from sqlalchemy.sql import func
from database.base import Base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MLModelDB(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer)
    algorithm_name = Column(String)
    model_path = Column(String)
    performance_metrics = Column(JSON)
    feature_names = Column(JSON)
    target_column = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    
    
class MLModel(BaseModel):
    dataset_id: int
    algorithm_name: str
    model_path: str
    performance_metrics: dict
    feature_names: list
    target_column: list
    created_at: datetime
    
    
    
class MLModelCreate(BaseModel):
    dataset_id: int
    algorithm_name: str
    model_path: str
    performance_metrics: dict
    target_column: list
    feature_names: list
    
    def create_db_instance(self):
        return MLModelDB(
            dataset_id=self.dataset_id,
            algorithm_name=self.algorithm_name,
            model_path=self.model_path,
            performance_metrics=self.performance_metrics,
            feature_names=self.feature_names,
            target_column=self.target_column
        )
        
        
class MLModelUpdate(BaseModel):
    dataset_id: Optional[int] = None
    algorithm_name: Optional[str] = None
    model_path: Optional[str] = None
    performance_metrics: Optional[dict] = None
    feature_names: Optional[list] = None
    target_column: Optional[list] = None
    created_at: Optional[datetime] = None
    
    
    def update_db_instance(self, db_ml_model: MLModelDB):
        if self.dataset_id:
            db_ml_model.dataset_id = self.dataset_id
        if self.algorithm_name:
            db_ml_model.algorithm_name = self.algorithm_name
        if self.model_path:
            db_ml_model.model_path = self.model_path
        if self.feature_names:
            db_ml_model.feature_names = self.feature_names
        if self.target_column:
            db_ml_model.target_column = self.target_column
        if self.performance_metrics:
            db_ml_model.performance_metrics = self.performance_metrics
        return db_ml_model