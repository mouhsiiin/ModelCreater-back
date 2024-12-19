from sqlalchemy import Column, Integer, String, JSON, DateTime, Boolean
from sqlalchemy.sql import func
from database.base import Base
from pydantic import BaseModel
from sqlalchemy.orm import relationship
from typing import Optional



class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    disabled = Column(Boolean, default=False)
    
    # relationships
    projects = relationship("ProjectDB", back_populates="owner")
    
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None


class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None



class UserCreate(User):
    password: str
    
    def create_db_instance(self):
        return UserDB(
            username=self.username,
            email=self.email,
            full_name=self.full_name,
            hashed_password=self.password,
            disabled=self.disabled
        )

class UserUpdate(User):
    password: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    email: Optional[str] = None
    username: Optional[str] = None
    
    def update_db_instance(self, db_user: UserDB):
        if self.password:
            db_user.hashed_password = self.password
        if self.full_name:
            db_user.full_name = self.full_name
        if self.disabled:
            db_user.disabled = self.disabled
        if self.email:
            db_user.email = self.email
        if self.username:
            db_user.username = self.username
        return db_user