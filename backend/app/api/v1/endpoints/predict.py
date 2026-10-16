import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Header, HTTPException, status, Depends, Request
from PIL import Image, UnidentifiedImageError

from backend.app.core.config import settings
from backend.app.core.security import get_current_user_optional
from backend.app.schemas.prediction import PredictionResponse, OmniPredictionResponse
from backend.app.services.classifier import classifier
from backend.app.services.database import db_service
from backend.app.services.gpt_service import gpt_service

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


@router.post("/predict", response_model=PredictionResponse, summary="Classify a waste image")
async def predict_waste_image(
    request: Request,
    file: UploadFile = File(..., description="Waste image file (JPG, PNG, WebP)"),
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Classifies an uploaded image into one of 6 supported categories:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash

    Returns the top prediction, confidence score, top-3 candidates,
    and actionable recycling guidelines.
    """
    # 1. Validate content type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a valid JPG, PNG, or WebP image."
        )

    # 2. Read bytes and check size
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded image file."
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds the maximum allowed limit of {settings.MAX_UPLOAD_MB}MB."
        )

    # 3. Decode image with PIL safely
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()  # Verify integrity
        # Reopen because verify() closes the stream
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, ValueError, OSError) as e:
        logger.warning(f"Corrupted or invalid image uploaded: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or decipherable image."
        )

    # 4. Perform Inference
    try:
        result = classifier.predict(image)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during model prediction. Please verify that the model is loaded."
        )

    # 5. Persist prediction record in database
    user_id = current_user.get("id") if current_user else None
    client_ip = request.client.host if request.client else "unknown"
    db_service.save_prediction(result, user_id=user_id)
    db_service.log_activity(
        action="PREDICTION_CREATED",
        details=f"Classified as {result.prediction} ({result.confidence_percentage}%)",
        user_id=user_id,
        ip_address=client_ip
    )

    return result
@router.post("/predict/omni", response_model=OmniPredictionResponse, summary="Universal waste classification via ChatGPT Vision")
async def predict_waste_omni(
    request: Request,
    file: UploadFile = File(..., description="Image file of any item, material, or waste (JPG, PNG, WebP)"),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key", description="Optional user-provided OpenAI API Key"),
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Universal AI Waste & Material Classifier powered by ChatGPT Vision (GPT-4o-mini).
    Identifies ANY item (including complex e-waste, clothing, batteries, medical waste, hazardous items)
    with deep material breakdown, recyclability, and safety protocols.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a valid JPG, PNG, or WebP image."
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded image file."
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds the maximum allowed limit of {settings.MAX_UPLOAD_MB}MB."
        )

    # Decode and verify with PIL
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except (UnidentifiedImageError, ValueError, OSError) as e:
        logger.warning(f"Corrupted or invalid image uploaded: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid or decipherable image."
        )

    # Perform Omni Inference via GPT Service
    result = gpt_service.analyze_image_omni(content, user_api_key=x_openai_key)

    # Persist in DB & activity log
    user_id = current_user.get("id") if current_user else None
    client_ip = request.client.host if request.client else "unknown"
    db_service.save_omni_prediction(result, user_id=user_id)
    db_service.log_activity(
        action="OMNI_PREDICTION_CREATED",
        details=f"Omni-Classified '{result.item_name}' as {result.category} ({result.confidence_percentage}%)",
        user_id=user_id,
        ip_address=client_ip
    )

    return result

