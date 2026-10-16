import os
import psutil
from datetime import datetime, timedelta
import random
import uuid
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.security import verify_admin_access
from backend.app.core.config import settings
from backend.app.services.classifier import classifier
from backend.app.services.database import db_service
from backend.app.schemas.prediction import PredictionResponse, PredictionItem, RecyclingGuidance
from backend.app.services.classifier import RECYCLING_GUIDES

router = APIRouter()

@router.get("/stats", summary="Get administrative system metrics and health")
def get_admin_system_stats(admin_user: dict = Depends(verify_admin_access)):
    """
    Returns system telemetry, resource consumption, model loading status,
    and storage health for administrators.
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    analytics = db_service.get_analytics()

    return {
        "system_status": "operational",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "authenticated_admin": admin_user.get("id"),
        "auth_type": admin_user.get("type"),
        "model": {
            "loaded": classifier.is_loaded,
            "architecture": "MobileNetV2 Transfer Learning",
            "classes": settings.CLASSES,
            "parameters": 2422726,
            "version": settings.VERSION
        },
        "database": {
            "mode": "Supabase" if db_service.supabase_client else "SQLite Local Fallback",
            "total_records": analytics.total_predictions,
            "storage_path": settings.SQLITE_DB_PATH
        },
        "resources": {
            "rss_memory_mb": round(memory_info.rss / (1024 * 1024), 2),
            "vms_memory_mb": round(memory_info.vms / (1024 * 1024), 2),
            "cpu_percent": psutil.cpu_percent(interval=None)
        }
    }

@router.get("/logs", summary="Get system audit and security logs")
def get_admin_logs(
    limit: int = 50,
    admin_user: dict = Depends(verify_admin_access)
):
    """
    Fetches the latest activity audit logs for system operations.
    """
    return db_service.get_activity_logs(limit=limit)

@router.post("/seed-demo", summary="Seed demonstration records if database is empty")
def seed_demo_data(admin_user: dict = Depends(verify_admin_access)):
    """
    Seeds a representative history of realistic prediction records across
    the last 7 days to facilitate viva presentations and portfolio reviews.
    Only seeds if fewer than 5 records exist.
    """
    current_analytics = db_service.get_analytics()
    if current_analytics.total_predictions >= 5:
        return {
            "status": "skipped",
            "message": f"Database already contains {current_analytics.total_predictions} records. Seeding not required."
        }

    classes = settings.CLASSES
    sample_counts = {
        "plastic": 18,
        "paper": 15,
        "cardboard": 12,
        "metal": 10,
        "glass": 9,
        "trash": 5
    }

    seeded = 0
    now = datetime.utcnow()

    for category, count in sample_counts.items():
        guide_dict = RECYCLING_GUIDES.get(category, RECYCLING_GUIDES["trash"])
        for i in range(count):
            days_ago = random.randint(0, 10)
            record_time = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            conf = round(random.uniform(0.78, 0.99), 4)

            # Pick 2 other classes for top 3
            others = [c for c in classes if c != category]
            c2, c3 = random.sample(others, 2)
            s2 = round(random.uniform(0.01, (1.0 - conf) * 0.7), 4)
            s3 = round(1.0 - conf - s2, 4)

            top_preds = [
                PredictionItem(class_name=category, score=conf, percentage=round(conf * 100, 2)),
                PredictionItem(class_name=c2, score=s2, percentage=round(s2 * 100, 2)),
                PredictionItem(class_name=c3, score=s3, percentage=round(s3 * 100, 2))
            ]

            guidance = RecyclingGuidance(
                category=category,
                is_recyclable=guide_dict["is_recyclable"],
                summary=guide_dict["summary"],
                common_examples=guide_dict["common_examples"],
                disposal_guide=guide_dict["disposal_guide"],
                recycling_tips=guide_dict["recycling_tips"],
                contamination_warnings=guide_dict["contamination_warnings"]
            )

            pred_resp = PredictionResponse(
                id=str(uuid.uuid4()),
                prediction=category,
                confidence=conf,
                confidence_percentage=round(conf * 100, 2),
                top_predictions=top_preds,
                guidance=guidance,
                model_version=settings.VERSION,
                inference_time_ms=round(random.uniform(25.0, 60.0), 2),
                timestamp=record_time.isoformat() + "Z"
            )

            db_service.save_prediction(pred_resp, user_id="demo-user")
            seeded += 1

    db_service.log_activity("DEMO_DATA_SEEDED", f"Seeded {seeded} realistic prediction samples", user_id="admin")

    return {
        "status": "success",
        "records_seeded": seeded,
        "message": f"Successfully populated {seeded} demo classification records."
    }
