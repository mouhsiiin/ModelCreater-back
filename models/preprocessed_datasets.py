from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base


class PreprocessedDatasetDB(Base):
    __tablename__ = "preprocessed_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    config_id = Column(Integer, ForeignKey("preprocessing_configurations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    location = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")  # pending, completed, failed
    pre_metadata = Column(JSON, nullable=True)  # Store additional preprocessing metadata
    notes = Column(Text)

    # Relationships
    dataset = relationship("DatasetDB", back_populates="preprocessed_datasets")
    config = relationship("PreprocessingConfigurationDB", back_populates="preprocessed_datasets")
