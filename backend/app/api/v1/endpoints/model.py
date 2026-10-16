from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.model import ModelInfoResponse, ModelMetricsResponse
from backend.app.services.classifier import classifier
from backend.app.services.metrics import metrics_service

router = APIRouter()

@router.get("/info", response_model=ModelInfoResponse, summary="Get model architecture and metadata")
def get_model_info():
    """
    Returns structural details about the trained classifier,
    including backbone network, input dimensions, parameter counts,
    and class label mapping.
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model is not currently loaded into memory."
        )
    return classifier.get_model_summary_info()

@router.get("/metrics", response_model=ModelMetricsResponse, summary="Get verified model evaluation metrics")
def get_model_metrics():
    """
    Returns verified evaluation metrics computed on the test dataset:
    - Overall test accuracy
    - Precision, recall, and F1 scores (macro and weighted)
    - Confusion matrix (6x6)
    - Per-class performance breakdown
    - Training and validation learning curves
    """
    metrics = metrics_service.load_metrics()
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics have not been generated yet. Run scripts/run_evaluation.py to produce evaluation reports."
        )
    return metrics
