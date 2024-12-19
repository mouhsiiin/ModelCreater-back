from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from database.base import Base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime 




class DatasetDB(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String)
    dataset_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class Dataset(BaseModel):
    name: str
    file_path: str
    dataset_metadata: dict
    created_at: datetime
    
    
    
class DatasetCreate(BaseModel):
    name: str
    file_path: str
    dataset_metadata: dict
    
    def create_db_instance(self):
        return DatasetDB(
            name=self.name,
            file_path=self.file_path,
            dataset_metadata=self.dataset_metadata
        )
    
class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    file_path: Optional[str] = None
    dataset_metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    
    
    def update_db_instance(self, db_dataset: DatasetDB):
        if self.name:
            db_dataset.name = self.name
        if self.file_path:
            db_dataset.file_path = self.file_path
        if self.dataset_metadata:
            db_dataset.dataset_metadata = self.dataset_metadata
        return db_dataset
    