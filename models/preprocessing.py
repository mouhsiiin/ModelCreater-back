from sqlalchemy import Column, Integer, String, JSON
from database.base import Base
from pydantic import BaseModel
from sqlalchemy.orm import relationship



class PreprocessingConfigurationDB(Base):
    __tablename__ = "preprocessing_configurations"
    
    id = Column(Integer, primary_key=True)
    options = Column(JSON, nullable=False)
    preview_stats = Column(JSON, nullable=False)
    
    # Relationships
    preprocessed_datasets = relationship("PreprocessedDatasetDB", back_populates="config")

    
    


class PreprocessingConfiguration(BaseModel):
    config_id: str
    options: dict
    preview_stats: dict
    
    class Config:
        orm_mode = True
        
        
        
        
class PreprocessingConfigurationCreate(BaseModel):
    options: dict
    preview_stats: dict
    
    class Config:
        orm_mode = True