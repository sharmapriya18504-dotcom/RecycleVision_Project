import io
import csv
from typing import Optional
from fastapi import APIRouter, Query, HTTPException, status, Response, Depends
from fastapi.responses import StreamingResponse

from backend.app.schemas.prediction import PredictionListResponse, PredictionRecord
from backend.app.services.database import db_service
from backend.app.core.security import get_current_user_optional

router = APIRouter()

@router.get("", response_model=PredictionListResponse, summary="List past predictions with filters and pagination")
def list_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by predicted category"),
    search: Optional[str] = Query(None, description="Search by ID or class name"),
    start_date: Optional[str] = Query(None, description="Filter created on or after (ISO date)"),
    end_date: Optional[str] = Query(None, description="Filter created on or before (ISO date)"),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Returns a paginated list of classification events.
    Ordinary users only see their own predictions if authenticated,
    or general guest records if unauthenticated.
    """
    user_id = current_user.get("id") if (current_user and current_user.get("role") != "admin") else None
    
    items, total = db_service.list_predictions(
        page=page,
        limit=limit,
        category=category,
        search=search,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id
    )

    pages = (total + limit - 1) // limit if total > 0 else 1

    return PredictionListResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        pages=pages
    )

@router.get("/export/csv", summary="Export prediction history as CSV")
def export_predictions_csv(
    category: Optional[str] = Query(None),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Generates and streams a CSV export of prediction history.
    """
    user_id = current_user.get("id") if (current_user and current_user.get("role") != "admin") else None
    items, _ = db_service.list_predictions(
        page=1,
        limit=5000,
        category=category,
        user_id=user_id
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Prediction ID",
        "Predicted Class",
        "Confidence Score",
        "Confidence Percentage",
        "Top 1 Candidate",
        "Top 1 Score",
        "Top 2 Candidate",
        "Top 2 Score",
        "Top 3 Candidate",
        "Top 3 Score",
        "Model Version",
        "Timestamp"
    ])

    for item in items:
        top1 = item.top_predictions[0].class_name if len(item.top_predictions) > 0 else ""
        top1_s = item.top_predictions[0].score if len(item.top_predictions) > 0 else ""
        top2 = item.top_predictions[1].class_name if len(item.top_predictions) > 1 else ""
        top2_s = item.top_predictions[1].score if len(item.top_predictions) > 1 else ""
        top3 = item.top_predictions[2].class_name if len(item.top_predictions) > 2 else ""
        top3_s = item.top_predictions[2].score if len(item.top_predictions) > 2 else ""

        writer.writerow([
            item.id,
            item.predicted_class,
            item.confidence,
            f"{item.confidence * 100:.2f}%",
            top1, top1_s,
            top2, top2_s,
            top3, top3_s,
            item.model_version,
            item.created_at
        ])

    output.seek(0)
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=recyclevision_prediction_history.csv"
    return response

@router.get("/{prediction_id}", response_model=PredictionRecord, summary="Get individual prediction detail")
def get_prediction_detail(prediction_id: str):
    items, _ = db_service.list_predictions(page=1, limit=1, search=prediction_id)
    if not items or items[0].id != prediction_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID '{prediction_id}' was not found."
        )
    return items[0]
