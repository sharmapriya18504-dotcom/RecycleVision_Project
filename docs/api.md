# RecycleVision AI — REST API Documentation (v1)

**Base URL**: `http://localhost:8000/api/v1` (Development)  
**Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)  
**ReDoc**: `http://localhost:8000/redoc`  

---

## 1. Overview of Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/health` | System health and model loading status | Public |
| `GET` | `/api/v1/model/info` | Architecture, classes, parameter count | Public |
| `GET` | `/api/v1/model/metrics` | Verified test accuracy, confusion matrix, learning curves | Public |
| `POST` | `/api/v1/predict` | Upload image for garbage classification (MobileNetV2 Core) | Public (Guest or Auth) |
| `POST` | `/api/v1/predict/omni` | Universal waste classification via ChatGPT Vision (ANY Item) | Public (Optional `X-OpenAI-Key`) |
| `POST` | `/api/v1/chat` | Conversational recycling assistant (RecycleBot GPT-4o) | Public (Optional `X-OpenAI-Key`) |
| `GET` | `/api/v1/predictions` | Paginated prediction history with filters | Public / User owned |
| `GET` | `/api/v1/predictions/{id}`| Detail of a specific prediction event | Public / User owned |
| `GET` | `/api/v1/predictions/export/csv`| Stream prediction history as downloadable CSV | Public / User owned |
| `GET` | `/api/v1/analytics` | Category distribution, daily trends, aggregate stats | Public |
| `GET` | `/api/v1/admin/stats` | RAM, CPU, DB connection, operational telemetry | Admin (`X-Admin-Key` or Admin JWT) |
| `GET` | `/api/v1/admin/logs` | Audit trail of administrative actions | Admin |
| `POST` | `/api/v1/admin/seed-demo` | Seeds demo predictions for demonstration / viva | Admin |


---

## 2. Endpoint Details

### 2.1 Health Check
- **Endpoint**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "RecycleVision AI",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2026-10-16T15:10:00Z"
}
```

---

### 2.2 Model Architecture Info
- **Endpoint**: `GET /api/v1/model/info`
- **Response**:
```json
{
  "model_name": "RecycleVision MobileNetV2 Classifier",
  "architecture": "MobileNetV2 Transfer Learning with GlobalAveragePooling2D",
  "base_model": "MobileNetV2 (ImageNet weights, frozen backbone)",
  "input_shape": [null, 224, 224, 3],
  "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "num_classes": 6,
  "total_parameters": 2422726,
  "trainable_parameters": 164742,
  "non_trainable_parameters": 2257984,
  "normalization": "1/255 rescale (pixel values in [0, 1])",
  "activation": "Softmax",
  "model_version": "1.0.0",
  "status": "ready"
}
```

---

### 2.3 Model Evaluation Metrics
- **Endpoint**: `GET /api/v1/model/metrics`
- **Response**:
```json
{
  "status": "evaluated",
  "test_accuracy": 0.7853,
  "macro_precision": 0.7612,
  "macro_recall": 0.7498,
  "macro_f1": 0.7522,
  "weighted_f1": 0.7810,
  "evaluated_samples": 503,
  "class_metrics": {
    "cardboard": {"precision": 0.88, "recall": 0.84, "f1_score": 0.86, "support": 80},
    "glass": {"precision": 0.75, "recall": 0.78, "f1_score": 0.76, "support": 100},
    "metal": {"precision": 0.72, "recall": 0.73, "f1_score": 0.72, "support": 82},
    "paper": {"precision": 0.85, "recall": 0.88, "f1_score": 0.86, "support": 118},
    "plastic": {"precision": 0.71, "recall": 0.69, "f1_score": 0.70, "support": 96},
    "trash": {"precision": 0.65, "recall": 0.59, "f1_score": 0.62, "support": 27}
  },
  "confusion_matrix": {
    "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "matrix": [[...], [...], ...]
  },
  "training_history": [
    {"epoch": 1, "train_accuracy": 0.621, "val_accuracy": 0.715, "train_loss": 1.15, "val_loss": 0.92},
    ...
  ],
  "evaluation_dataset": "TrashNet 20% Held-Out Validation Split",
  "evaluated_at": "2026-10-16T15:15:52Z"
}
```

---

### 2.4 Predict Waste Image
- **Endpoint**: `POST /api/v1/predict`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (Binary image file: JPG, PNG, WebP. Max 10MB)
- **Response**:
```json
{
  "id": "e44d320b-60a5-4eb4-9ae8-53b7cba7bc12",
  "prediction": "cardboard",
  "confidence": 0.9624,
  "confidence_percentage": 96.24,
  "top_predictions": [
    {"class_name": "cardboard", "score": 0.9624, "percentage": 96.24},
    {"class_name": "paper", "score": 0.0315, "percentage": 3.15},
    {"class_name": "trash", "score": 0.0061, "percentage": 0.61}
  ],
  "guidance": {
    "category": "cardboard",
    "is_recyclable": true,
    "summary": "Cardboard and corrugated boxes are highly recyclable when clean and dry.",
    "common_examples": ["Shipping boxes", "Cereal boxes", "Shoe boxes"],
    "disposal_guide": "Flatten all boxes to save space in the recycling bin...",
    "recycling_tips": ["Flatten boxes completely before putting in cart..."],
    "contamination_warnings": ["Pizza boxes soaked with heavy grease cannot be recycled..."],
    "local_rules_note": "Recycling capabilities vary significantly by municipality."
  },
  "model_version": "1.0.0",
  "inference_time_ms": 32.45,
  "timestamp": "2026-10-16T15:20:00Z"
}
```

---

### 2.5 Prediction History & Pagination
- **Endpoint**: `GET /api/v1/predictions?page=1&limit=20&category=plastic&search=bottle`
- **Response**:
```json
{
  "items": [
    {
      "id": "...",
      "user_id": null,
      "predicted_class": "plastic",
      "confidence": 0.8995,
      "top_predictions": [...],
      "model_version": "1.0.0",
      "created_at": "2026-10-15T19:36:39Z"
    }
  ],
  "total": 70,
  "page": 1,
  "limit": 20,
  "pages": 4
}
```

---

### 2.6 Export Prediction History (CSV)
- **Endpoint**: `GET /api/v1/predictions/export/csv`
- **Response**: File stream (`Content-Disposition: attachment; filename=recyclevision_prediction_history.csv`)

---

### 2.7 Analytics Summary
- **Endpoint**: `GET /api/v1/analytics`
- **Response**:
```json
{
  "total_predictions": 70,
  "category_distribution": [
    {"category": "plastic", "count": 18, "percentage": 25.71},
    {"category": "paper", "count": 15, "percentage": 21.43},
    {"category": "cardboard", "count": 12, "percentage": 17.14},
    {"category": "glass", "count": 10, "percentage": 14.29},
    {"category": "metal", "count": 10, "percentage": 14.29},
    {"category": "trash", "count": 5, "percentage": 7.14}
  ],
  "most_frequent_category": "plastic",
  "daily_trends": [
    {"date": "2026-10-14", "count": 7},
    {"date": "2026-10-15", "count": 9},
    {"date": "2026-10-16", "count": 4}
  ],
  "average_confidence": 0.8753,
  "recent_activity": [...]
}
```

---

### 2.8 Administrative Health & Telemetry
- **Endpoint**: `GET /api/v1/admin/stats`
- **Headers**:
  - `X-Admin-Key: recyclevision-admin-dev-secret-key-2026`
- **Response**:
```json
{
  "system_status": "operational",
  "authenticated_admin": "admin-system",
  "model": {
    "loaded": true,
    "architecture": "MobileNetV2 Transfer Learning",
    "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "parameters": 2422726,
    "version": "1.0.0"
  },
  "database": {
    "mode": "SQLite Local Fallback",
    "total_records": 70
  },
  "resources": {
    "rss_memory_mb": 245.32,
    "cpu_percent": 3.4
  }
}
```
