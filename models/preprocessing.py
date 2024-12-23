from sqlalchemy import Column, Integer, String, JSON
from database.base import Base
from pydantic import BaseModel



class PreprocessingConfigurationDB(Base):
    __tablename__ = "preprocessing_configurations"
    
    id = Column(Integer, primary_key=True)
    config_id = Column(String, unique=True, nullable=False)
    options = Column(JSON, nullable=False)
    preview_stats = Column(JSON, nullable=False)
    
    # Relationships
    
    


class PreprocessingConfiguration(BaseModel):
    config_id: str
    options: dict
    preview_stats: dict
    
    class Config:
        orm_mode = True