from sqlalchemy import Column, Integer, String, JSON, DateTime, Boolean
from sqlalchemy.sql import func
from .base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    file_path = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    model_metadata = Column(JSON)
    
    
class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer)
    algorithm_name = Column(String)
    model_path = Column(String)
    performance_metrics = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    
    