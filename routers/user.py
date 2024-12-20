from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.users import UserCreate, UserDB, User
from database.base import get_db
from security.auth import get_current_user
from passlib.context import CryptContext

# Initialize router
router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    """Hash the user's password"""
    return pwd_context.hash(password)

@router.post("/register", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user in the database
    
    - Checks if username already exists
    - Hashes the password
    - Saves the new user to the database
    """
    # Check if user already exists
    existing_user = db.query(UserDB).filter(UserDB.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create new user model
    new_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )
    
    # Add and commit to database
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


# get all users
@router.get("/all", response_model=list[User])
def get_all_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Get all users in the database
    
    - Requires a valid JWT token
    - Returns a list of users
    """
    return db.query(UserDB).all()


@router.get("/profile", response_model=User)
def get_user_profile(current_user: User = Depends(get_current_user)):
    """
    Get the current user's profile information
    
    - Requires a valid JWT token
    - Returns user details
    """
    return current_user

@router.put("/profile", response_model=User)
def update_user_profile(
    user_update: UserCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """
    Update the current user's profile
    
    - Requires a valid JWT token
    - Allows updating email and password
    """
    # Update user details
    current_user.email = user_update.email
    
    # Update password if provided
    if user_update.password:
        current_user.hashed_password = hash_password(user_update.password)
    
    # Commit changes
    db.commit()
    db.refresh(current_user)
    
    return current_user