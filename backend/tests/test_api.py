import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.core.config import settings

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data

def test_model_info_endpoint():
    response = client.get(f"{settings.API_V1_STR}/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 6
    assert "MobileNetV2" in data["architecture"]
    assert data["classes"] == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def test_predict_endpoint_valid_image():
    # Create test image in memory
    img = Image.new("RGB", (224, 224), color=(0, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        f"{settings.API_V1_STR}/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in settings.CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 3
    assert "guidance" in data
    assert "summary" in data["guidance"]

def test_predict_endpoint_invalid_content_type():
    buf = io.BytesIO(b"dummy text file content")
    response = client.post(
        f"{settings.API_V1_STR}/predict",
        files={"file": ("test.txt", buf, "text/plain")}
    )
    assert response.status_code == 415

def test_predict_endpoint_corrupted_image():
    buf = io.BytesIO(b"not an image binary payload")
    response = client.post(
        f"{settings.API_V1_STR}/predict",
        files={"file": ("fake.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 400

def test_analytics_endpoint():
    response = client.get(f"{settings.API_V1_STR}/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "category_distribution" in data
    assert isinstance(data["category_distribution"], list)

def test_predictions_history_endpoint():
    response = client.get(f"{settings.API_V1_STR}/predictions")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "pages" in data

def test_predictions_csv_export():
    response = client.get(f"{settings.API_V1_STR}/predictions/export/csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "Prediction ID,Predicted Class" in response.text

def test_admin_stats_unauthorized():
    response = client.get(f"{settings.API_V1_STR}/admin/stats")
    assert response.status_code == 403

def test_admin_stats_authorized():
    response = client.get(
        f"{settings.API_V1_STR}/admin/stats",
        headers={"X-Admin-Key": settings.ADMIN_SECRET_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["system_status"] == "operational"
    assert "resources" in data

def test_predict_omni_endpoint():
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        f"{settings.API_V1_STR}/predict/omni",
        files={"file": ("omni_test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "item_name" in data
    assert "category" in data
    assert "is_recyclable" in data
    assert "disposal_instructions" in data
    assert isinstance(data["disposal_instructions"], list)

def test_chat_endpoint():
    response = client.post(
        f"{settings.API_V1_STR}/chat",
        json={"messages": [{"role": "user", "content": "Can I recycle aluminum cans?"}]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "model" in data

