import os
import json
import logging
from typing import Optional, Dict, Any
from backend.app.core.config import settings
from backend.app.schemas.model import (
    ModelMetricsResponse,
    ClassMetric,
    ConfusionMatrixData,
    TrainingEpochMetric
)

logger = logging.getLogger(__name__)

class MetricsService:
    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)

    def get_metrics_filepath(self) -> str:
        candidates = [
            os.path.join(self.reports_dir, "metrics.json"),
            os.path.join(os.getcwd(), "backend", "reports", "metrics.json"),
            os.path.join(os.getcwd(), "reports", "metrics.json"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "metrics.json")),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def load_metrics(self) -> Optional[ModelMetricsResponse]:
        filepath = self.get_metrics_filepath()
        if not os.path.exists(filepath):
            logger.warning(f"Metrics file not found at {filepath}")
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            class_metrics = {}
            for k, v in data.get("class_metrics", {}).items():
                class_metrics[k] = ClassMetric(
                    precision=v.get("precision", 0.0),
                    recall=v.get("recall", 0.0),
                    f1_score=v.get("f1_score", 0.0),
                    support=v.get("support", 0)
                )

            cm_data = data.get("confusion_matrix", {})
            confusion_matrix = ConfusionMatrixData(
                classes=cm_data.get("classes", settings.CLASSES),
                matrix=cm_data.get("matrix", [])
            )

            history = []
            for h in data.get("training_history", []):
                history.append(TrainingEpochMetric(
                    epoch=h.get("epoch", 1),
                    train_accuracy=h.get("train_accuracy", 0.0),
                    val_accuracy=h.get("val_accuracy", 0.0),
                    train_loss=h.get("train_loss", 0.0),
                    val_loss=h.get("val_loss", 0.0)
                ))

            return ModelMetricsResponse(
                status="evaluated",
                test_accuracy=data.get("test_accuracy", 0.0),
                macro_precision=data.get("macro_precision", 0.0),
                macro_recall=data.get("macro_recall", 0.0),
                macro_f1=data.get("macro_f1", 0.0),
                weighted_f1=data.get("weighted_f1", 0.0),
                evaluated_samples=data.get("evaluated_samples", 0),
                class_metrics=class_metrics,
                confusion_matrix=confusion_matrix,
                training_history=history,
                evaluation_dataset=data.get("evaluation_dataset", "TrashNet 20% Held-Out Validation Split"),
                evaluated_at=data.get("evaluated_at", ""),
                known_limitations=data.get("known_limitations", [
                    "Single-object classification: Designed for clean, center-focused waste images, not cluttered multi-item scenes.",
                    "Lighting & reflection: High glare on glass or translucent plastic cups may cause misclassification between glass and plastic.",
                    "Heavily compressed or contaminated materials: Crushed or soiled cans and cardboard may exhibit higher prediction uncertainty.",
                    "Class imbalance in training data: Trash class contains fewer original samples (137) compared to paper (594)."
                ])
            )
        except Exception as e:
            logger.error(f"Error parsing metrics JSON: {e}")
            return None

metrics_service = MetricsService()
