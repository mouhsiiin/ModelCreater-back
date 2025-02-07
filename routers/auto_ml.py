from fastapi import APIRouter, File, HTTPException, UploadFile

# Assume UPLOAD_DIRECTORY is defined in your routers/dataset module.
from routers.dataset import UPLOAD_DIRECTORY
from services.auto_model_crafter import AutoModelCrafter

router = APIRouter(prefix="/auto", tags=["auto_Crafter"])

@router.post("/craft")
async def auto_ml_pipeline(file: UploadFile = File(...)):
    """
    Endpoint that receives a file upload, then uses the AutoModelCrafter class
    to process the file, select the best model (with hyperparameter tuning),
    and return the evaluation results.
    """
    try:
        crafter = AutoModelCrafter(file)
        result = await crafter.craft_model(UPLOAD_DIRECTORY)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))