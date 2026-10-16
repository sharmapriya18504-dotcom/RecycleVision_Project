from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ClassMetric(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int

class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    base_model: str
    input_shape: List[Optional[int]]
    classes: List[str]
    num_classes: int
    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int
    normalization: str
    activation: str
    model_version: str
    status: str

class ConfusionMatrixData(BaseModel):
    classes: List[str]
    matrix: List[List[int]]

class TrainingEpochMetric(BaseModel):
    epoch: int
    train_accuracy: float
    val_accuracy: float
    train_loss: float
    val_loss: float

class ModelMetricsResponse(BaseModel):
    status: str
    test_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    evaluated_samples: int
    class_metrics: Dict[str, ClassMetric]
    confusion_matrix: ConfusionMatrixData
    training_history: List[TrainingEpochMetric]
    evaluation_dataset: str
    evaluated_at: str
    known_limitations: List[str]
