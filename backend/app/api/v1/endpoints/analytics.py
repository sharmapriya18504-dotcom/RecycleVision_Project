from fastapi import APIRouter
from backend.app.schemas.analytics import AnalyticsResponse
from backend.app.services.database import db_service

router = APIRouter()

@router.get("", response_model=AnalyticsResponse, summary="Get aggregate waste classification analytics")
def get_analytics():
    """
    Computes real-time analytics from stored prediction records:
    - Total predictions recorded
    - Class distribution and percentages
    - Most commonly classified waste type
    - Daily prediction trends over time
    - Average prediction confidence
    - Recent prediction activity stream
    """
    return db_service.get_analytics()
