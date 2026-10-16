import os
import time
import uuid
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    PredictionResponse,
    PredictionItem,
    RecyclingGuidance
)

logger = logging.getLogger(__name__)

# Category-specific authentic recycling knowledge base
RECYCLING_GUIDES: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "is_recyclable": True,
        "summary": "Cardboard and corrugated boxes are highly recyclable when clean and dry.",
        "common_examples": ["Shipping boxes", "Cereal boxes", "Corrugated packaging", "Shoe boxes", "Egg cartons (clean)"],
        "disposal_guide": "Flatten all boxes to save space in the recycling bin. Remove plastic packing tape, styrofoam inserts, and shipping wrap before disposal.",
        "recycling_tips": [
            "Flatten boxes completely before putting them in the recycling cart.",
            "Remove all plastic bubble wrap, polystyrene packing peanuts, and excessive plastic tape.",
            "Small amounts of parcel tape are generally separated by pulping facilities, but heavy plastic liners must be stripped."
        ],
        "contamination_warnings": [
            "Pizza boxes soaked with heavy grease, cheese, or food debris cannot be recycled with cardboard (compost if clean or discard as trash).",
            "Wax-coated, wet, or water-damaged boxes weaken paper pulping fibers and ruin recycling batches."
        ]
    },
    "glass": {
        "is_recyclable": True,
        "summary": "Glass bottles and jars are 100% recyclable infinitely without loss of purity or quality.",
        "common_examples": ["Beverage bottles", "Condiment jars", "Jam and honey jars", "Cosmetic containers (glass)"],
        "disposal_guide": "Rinse containers with water to remove food residues. Separate metal or plastic lids if your local program requests separate lid processing.",
        "recycling_tips": [
            "A quick rinse is usually sufficient; soap is not required.",
            "Labels do not need to be scraped off; industrial glass recycling kilns burn them off cleanly.",
            "Sort by color (clear, green, amber) if your regional facility requires color segregation."
        ],
        "contamination_warnings": [
            "Do NOT place window panes, mirrors, drinking glasses, ceramics, Pyrex, or lightbulbs in standard recycling bins.",
            "Treated glassware has a different melting point than container glass and can shatter furnaces."
        ]
    },
    "metal": {
        "is_recyclable": True,
        "summary": "Aluminum and steel cans are among the most energy-efficient and recyclable consumer materials.",
        "common_examples": ["Soda cans", "Food tins and soup cans", "Aluminum foil & trays (clean)", "Aerosol cans (empty)"],
        "disposal_guide": "Empty and lightly rinse to prevent odors and pests. Place metal lids inside the cans or squeeze the rim slightly so lids do not fall out during sorting.",
        "recycling_tips": [
            "Crushing aluminum beverage cans saves significant volume in household bins.",
            "Steel and tin soup cans can be recycled endlessly into car parts, construction beams, and new containers.",
            "Ensure aerosol cans are completely depressurized and empty."
        ],
        "contamination_warnings": [
            "Paint cans, motor oil canisters, and hazardous chemical containers must be brought to specialized household hazard depots.",
            "Aluminum foil caked in baked cheese or food waste must be thrown in trash."
        ]
    },
    "paper": {
        "is_recyclable": True,
        "summary": "Office paper, newspapers, books, and brochures can be pulped and made into new paper products.",
        "common_examples": ["Newspapers & magazines", "Office printer paper", "Notebooks", "Envelopes (plastic windows removed if possible)", "Brochures"],
        "disposal_guide": "Keep paper clean, dry, and unsoiled. Place directly into your dry recycling bin.",
        "recycling_tips": [
            "Do not shred paper unless security requires it; shredded fibers are too short to bind into standard paper.",
            "Paperclips and staples are usually removed by industrial magnetic separators, but removing plastic bindings is recommended.",
            "Keep separate from wet kitchen waste."
        ],
        "contamination_warnings": [
            "Used paper towels, napkins, facial tissues, and greasy food wrapping paper CANNOT be recycled.",
            "Thermal cash register receipts often contain BPA/BPS chemicals and belong in the trash."
        ]
    },
    "plastic": {
        "is_recyclable": True,
        "summary": "Rigid plastic bottles, jugs, and tubs are commonly accepted depending on the resin identification code (#1, #2, #5).",
        "common_examples": ["PET water bottles (#1)", "HDPE milk jugs & detergent bottles (#2)", "PP food containers & yogurt tubs (#5)"],
        "disposal_guide": "Empty all liquids, rinse clean, and reattach screw caps so caps do not slip through mechanical sorting screens.",
        "recycling_tips": [
            "Check the resin number (usually embossed inside a triangular arrows symbol on the base).",
            "PET (#1) and HDPE (#2) bottles enjoy the highest recycling recovery rates worldwide.",
            "Keep caps firmly screwed onto rinsed bottles."
        ],
        "contamination_warnings": [
            "Plastic carrier bags, bubble wrap, chip bags, and cling film tangle in sorting machinery and belong at specialized drop-off points, not curbside bins.",
            "Black plastic containers are often invisible to optical sorting equipment and end up in landfills."
        ]
    },
    "trash": {
        "is_recyclable": False,
        "summary": "Non-recyclable municipal solid waste, composite materials, and heavily contaminated items.",
        "common_examples": ["Food wrappers", "Broken ceramics", "Used hygiene products", "Styrofoam cups", "Multi-layer laminate packaging"],
        "disposal_guide": "Bag tightly and dispose into the designated landfill or general municipal waste bin to prevent environmental littering.",
        "recycling_tips": [
            "Reduce consumption by choosing reusable alternatives (cloth bags, refillable containers).",
            "Separate organic compostable materials if local composting programs exist.",
            "Do not 'wish-cycle' non-recyclables into the blue bin, as this contaminates good recyclables."
        ],
        "contamination_warnings": [
            "Placing general trash in recycling carts increases sorting costs and risks entire truckloads being diverted to landfill."
        ]
    }
}

class ClassifierService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _resolve_model_path(self) -> str:
        candidates = [
            settings.MODEL_PATH,
            os.path.join(os.getcwd(), "backend", "models", "best_model.h5"),
            os.path.join(os.getcwd(), "models", "best_model.h5"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_model.h5")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "best_model.h5")),
        ]
        for path in candidates:
            if os.path.exists(path):
                logger.info(f"Resolved model file at: {path}")
                return path
        raise FileNotFoundError(f"Trained model file not found in candidates: {candidates}")

    def _initialize_model(self):
        try:
            model_path = self._resolve_model_path()
            logger.info(f"Loading Keras model from {model_path}...")
            # Load without compiling optimizer if not training, for fast loading
            self._model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("RecycleVision AI model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model_summary_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"status": "unloaded", "error": "Model not loaded"}
        
        return {
            "model_name": "RecycleVision MobileNetV2 Classifier",
            "architecture": "MobileNetV2 Transfer Learning with GlobalAveragePooling2D",
            "base_model": "MobileNetV2 (ImageNet weights, frozen backbone)",
            "input_shape": list(self._model.input_shape),
            "classes": settings.CLASSES,
            "num_classes": len(settings.CLASSES),
            "total_parameters": int(self._model.count_params()),
            "trainable_parameters": int(sum([tf.keras.backend.count_params(w) for w in self._model.trainable_weights])),
            "non_trainable_parameters": int(sum([tf.keras.backend.count_params(w) for w in self._model.non_trainable_weights])),
            "normalization": "1/255 rescale (pixel values in [0, 1])",
            "activation": "Softmax",
            "model_version": settings.VERSION,
            "status": "ready"
        }

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Applies exact preprocessing used during model training:
        1. Convert image to RGB (stripping alpha channel if present)
        2. Resize to (224, 224) using bilinear interpolation
        3. Convert to float32 NumPy array and scale by 1.0 / 255.0
        4. Expand dimensions to (1, 224, 224, 3)
        """
        rgb_image = image.convert("RGB")
        resized_image = rgb_image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, image: Image.Image) -> PredictionResponse:
        if not self.is_loaded:
            raise RuntimeError("Model is not initialized or failed to load.")

        start_time = time.time()
        
        # Preprocessing
        input_tensor = self.preprocess_image(image)
        
        # Inference
        raw_predictions = self._model.predict(input_tensor, verbose=0)[0]
        
        inference_time = (time.time() - start_time) * 1000.0  # in ms
        
        # Map classes
        classes = settings.CLASSES
        top_indices = np.argsort(raw_predictions)[::-1]
        
        top_predictions: List[PredictionItem] = []
        for idx in top_indices[:3]:
            score = float(raw_predictions[idx])
            top_predictions.append(
                PredictionItem(
                    class_name=classes[idx],
                    score=round(score, 4),
                    percentage=round(score * 100.0, 2)
                )
            )

        best_idx = int(top_indices[0])
        primary_class = classes[best_idx]
        confidence_score = float(raw_predictions[best_idx])
        
        # Fetch knowledge guide
        guide_dict = RECYCLING_GUIDES.get(primary_class, RECYCLING_GUIDES["trash"])
        guidance = RecyclingGuidance(
            category=primary_class,
            is_recyclable=guide_dict["is_recyclable"],
            summary=guide_dict["summary"],
            common_examples=guide_dict["common_examples"],
            disposal_guide=guide_dict["disposal_guide"],
            recycling_tips=guide_dict["recycling_tips"],
            contamination_warnings=guide_dict["contamination_warnings"]
        )

        prediction_id = str(uuid.uuid4())
        timestamp_str = datetime.utcnow().isoformat() + "Z"

        return PredictionResponse(
            id=prediction_id,
            prediction=primary_class,
            confidence=round(confidence_score, 4),
            confidence_percentage=round(confidence_score * 100.0, 2),
            top_predictions=top_predictions,
            guidance=guidance,
            model_version=settings.VERSION,
            inference_time_ms=round(inference_time, 2),
            timestamp=timestamp_str
        )

# Global singleton
classifier = ClassifierService()
