import os
import pytest
import numpy as np
from PIL import Image

from backend.app.services.classifier import classifier
from backend.app.core.config import settings

def test_model_is_loaded():
    assert classifier.is_loaded is True
    summary = classifier.get_model_summary_info()
    assert summary["status"] == "ready"
    assert summary["num_classes"] == 6
    assert summary["classes"] == ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def test_image_preprocessing():
    # Create synthetic test image (100x150 RGBA)
    test_img = Image.new("RGBA", (100, 150), color=(255, 0, 0, 255))
    processed = classifier.preprocess_image(test_img)
    
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

def test_inference_synthetic_image():
    test_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = classifier.predict(test_img)
    
    assert result.prediction in settings.CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_predictions) == 3
    assert result.top_predictions[0].class_name == result.prediction
    assert result.guidance.category == result.prediction
    assert len(result.guidance.recycling_tips) > 0

def test_inference_real_dataset_sample():
    # Find a sample from dataset if available
    sample_path = os.path.join("dataset", "cardboard", "cardboard1.jpg")
    if os.path.exists(sample_path):
        img = Image.open(sample_path)
        result = classifier.predict(img)
        assert result.prediction == "cardboard"
        assert result.confidence > 0.8
