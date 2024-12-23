from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database.base import Base


class PreprocessedDataset(Base):
    __tablename__ = "preprocessed_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    config_id = Column(Integer, ForeignKey("preprocessing_configurations.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    location = Column(String, nullable=False)
    notes = Column(Text)

    # Relationships
    dataset = relationship("Dataset", back_populates="preprocessed_datasets")
    config = relationship("PreprocessingConfig", back_populates="preprocessed_datasets")

