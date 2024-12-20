from sqlalchemy import Column, Integer, String, JSON, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from database.base import Base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from sqlalchemy.orm import relationship


class ProjectDB(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    description = Column(String)
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # relationships
    owner = relationship("UserDB", back_populates="projects")
    

class Project(BaseModel):
    id : int
    name: str
    owner_id: int
    description: Optional[str] = None
    created_at: datetime
    status: str
    
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    status: Optional[str] = "active"
    
    def create_db_instance(self, owner_id: int):
        created_at = datetime.now()
        
        return ProjectDB(
            name=self.name,
            description=self.description,
            status=self.status,
            owner_id=owner_id,
            created_at=created_at
        )
    
    
class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    
    def update_db_instance(self, db_project: ProjectDB):
        if self.name:
            db_project.name = self.name
        if self.description:
            db_project.description = self.description
        if self.status:
            db_project.status = self.status
        db_project.created_at = datetime.now()
        
        return db_project