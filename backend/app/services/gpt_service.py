import io
import time
import uuid
import json
import base64
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from PIL import Image

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    OmniPredictionResponse,
    OmniMaterialItem,
    ChatMessage,
    ChatResponse
)

logger = logging.getLogger(__name__)

class GPTService:
    def __init__(self):
        self.default_model = settings.OPENAI_MODEL

    def _resolve_api_key(self, user_api_key: Optional[str] = None) -> Optional[str]:
        if user_api_key and user_api_key.strip():
            return user_api_key.strip()
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip() and not settings.OPENAI_API_KEY.startswith("your-"):
            return settings.OPENAI_API_KEY.strip()
        return None

    def analyze_image_omni(
        self,
        image_bytes: bytes,
        user_api_key: Optional[str] = None
    ) -> OmniPredictionResponse:
        """
        Classifies ANY waste item or object using ChatGPT Vision (GPT-4o-mini).
        Returns deep composition, recyclability, hazard warnings, and disposal instructions.
        Falls back to a graceful guidance response if no API key is configured.
        """
        start_time = time.time()
        prediction_id = f"omni_{uuid.uuid4().hex[:12]}"
        api_key = self._resolve_api_key(user_api_key)

        if not api_key:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.info("OpenAI API key not provided for Omni-Prediction. Returning setup guidance.")
            return OmniPredictionResponse(
                id=prediction_id,
                engine="chatgpt-vision-demo",
                item_name="Omni-Mode Ready (API Key Needed)",
                category="General / Universal",
                is_recyclable=True,
                confidence_score=0.98,
                confidence_percentage=98.0,
                summary=(
                    "ChatGPT Vision is enabled! To analyze ANY custom object (e-waste, batteries, textiles, medical waste, etc.) "
                    "with live OpenAI reasoning, please set OPENAI_API_KEY in backend/.env or enter your key in the top settings bar."
                ),
                materials=[
                    OmniMaterialItem(material="Multi-material composite", percentage=60, is_recyclable=False),
                    OmniMaterialItem(material="Recyclable packaging component", percentage=40, is_recyclable=True)
                ],
                disposal_instructions=[
                    "Add your OPENAI_API_KEY in backend/.env or in the top settings bar.",
                    "Upload any object: broken headphones, lithium batteries, cloth rags, tetra paks, or chemicals.",
                    "Receive instant AI material breakdown and safety protocols."
                ],
                special_warnings=[
                    "Live ChatGPT mode requires an OpenAI API key. Local MobileNetV2 (6 classes) remains 100% active and free!"
                ],
                environmental_impact=(
                    "Proper classification of complex, multi-material items prevents hazardous substances like heavy metals "
                    "and microplastics from entering municipal landfills."
                ),
                upcycling_ideas=[
                    "Test with local MobileNetV2 engine on the Classify page for 100% offline classification of Cardboard, Glass, Metal, Paper, Plastic, and Trash."
                ],
                model_used=self.default_model,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            # Compress and encode image to base64
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            
            # Resize if too large to conserve tokens and speed up inference
            max_dimension = 1024
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension))

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            system_prompt = (
                "You are an expert waste classification, circular economy, and material science AI for RecycleVision AI. "
                "Analyze the uploaded image of an object or waste material. "
                "You must accurately identify ANY item (e.g., smartphone battery, blister pack, plastic soda bottle, denim jeans, circuit board, fluorescent tube, styrofoam box, aluminum can). "
                "Determine if it is recyclable, identify its materials, give step-by-step disposal instructions, warn about hazards, and describe eco-impact. "
                "Return ONLY a valid JSON object with the following schema:\n"
                "{\n"
                '  "item_name": "Specific identified item name",\n'
                '  "category": "Broad category (e.g. e-waste, hazardous, plastic, paper, cardboard, glass, metal, textiles, organic, construction, mixed trash)",\n'
                '  "is_recyclable": true/false,\n'
                '  "confidence_percentage": 95.0,\n'
                '  "summary": "Short 2-sentence summary of the item and its recyclability status",\n'
                '  "materials": [{"material": "Plastic (PET)", "percentage": 85, "is_recyclable": true}],\n'
                '  "disposal_instructions": ["Step 1", "Step 2", "Step 3"],\n'
                '  "special_warnings": ["Safety or contamination warning if any"],\n'
                '  "environmental_impact": "Impact on landfill, carbon emissions, or water systems",\n'
                '  "upcycling_ideas": ["Creative reuse or repurposing idea"]\n'
                "}"
            )

            response = client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this waste item or object for proper recycling and disposal:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=0.2
            )

            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            materials = [
                OmniMaterialItem(
                    material=m.get("material", "Unknown material"),
                    percentage=m.get("percentage"),
                    is_recyclable=bool(m.get("is_recyclable", False))
                )
                for m in parsed.get("materials", [])
            ]

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            conf_pct = float(parsed.get("confidence_percentage", 92.0))

            return OmniPredictionResponse(
                id=prediction_id,
                engine="chatgpt-vision",
                item_name=parsed.get("item_name", "Identified Waste Item"),
                category=parsed.get("category", "General"),
                is_recyclable=bool(parsed.get("is_recyclable", False)),
                confidence_score=round(conf_pct / 100.0, 4),
                confidence_percentage=conf_pct,
                summary=parsed.get("summary", "Waste item analyzed with ChatGPT Vision."),
                materials=materials,
                disposal_instructions=parsed.get("disposal_instructions", ["Dispose in general municipal waste bin."]),
                special_warnings=parsed.get("special_warnings", []),
                environmental_impact=parsed.get("environmental_impact", "Proper sorting diverts recyclable materials from landfills."),
                upcycling_ideas=parsed.get("upcycling_ideas", []),
                model_used=self.default_model,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"Error calling ChatGPT Vision API: {e}", exc_info=True)
            return OmniPredictionResponse(
                id=prediction_id,
                engine="chatgpt-vision-fallback",
                item_name="Analysis Encountered An Issue",
                category="Unknown",
                is_recyclable=False,
                confidence_score=0.5,
                confidence_percentage=50.0,
                summary=f"Could not complete OpenAI vision classification: {str(e)[:150]}. Please verify your API key or network connection.",
                materials=[OmniMaterialItem(material="Unspecified", percentage=100, is_recyclable=False)],
                disposal_instructions=[
                    "Check your internet connection and verify your OpenAI API quota.",
                    "You can also use our local MobileNetV2 engine on the Classify page for 100% offline classification."
                ],
                special_warnings=["API communication error with OpenAI"],
                environmental_impact="Recycling helps protect nature and lower carbon emissions.",
                upcycling_ideas=[],
                model_used=self.default_model,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

    def chat_recyclebot(
        self,
        messages: List[ChatMessage],
        user_api_key: Optional[str] = None
    ) -> ChatResponse:
        """
        Provides conversational AI guidance on recycling rules, e-waste, hazardous materials,
        and zero-waste living. Responds in English or Hindi/Hinglish based on the user's prompt.
        """
        api_key = self._resolve_api_key(user_api_key)

        if not api_key:
            return ChatResponse(
                reply=(
                    "👋 **Namaste & Hello! I am RecycleBot**, your AI recycling companion.\n\n"
                    "To chat with me live powered by **ChatGPT (GPT-4o)**, please enter your **OpenAI API Key** "
                    "in the top settings bar or add `OPENAI_API_KEY=your_key` to `backend/.env`.\n\n"
                    "Once connected, you can ask me anything like:\n"
                    "- *'How do I safely dispose of swollen lithium batteries?'*\n"
                    "- *'Can pizza boxes with grease be recycled?'*\n"
                    "- *'Purane kapde aur e-waste kahan submit karein?'*\n"
                    "- *'What do the plastic recycling resin codes 1-7 mean?'*"
                ),
                model=self.default_model,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system_instruction = (
                "You are RecycleBot, an expert, enthusiastic, and highly knowledgeable sustainability and waste management AI assistant "
                "for RecycleVision AI. You help users understand how to properly recycle, dispose of, or upcycle ANY item (including "
                "complex e-waste, hazardous chemicals, batteries, textiles, medical waste, and municipal solid waste). "
                "You are fluent in both English and Hindi/Hinglish. If the user asks in Hindi or Hinglish, respond warmly in natural Hinglish or Hindi. "
                "Always format answers with neat markdown bullet points, clear actionable steps, and safety warnings for toxic or hazardous items."
            )

            formatted_messages = [{"role": "system", "content": system_instruction}]
            for msg in messages:
                formatted_messages.append({"role": msg.role, "content": msg.content})

            response = client.chat.completions.create(
                model=self.default_model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=600
            )

            reply_text = response.choices[0].message.content or "I couldn't generate a response. Please try again."

            return ChatResponse(
                reply=reply_text,
                model=self.default_model,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        except Exception as e:
            logger.error(f"Error in RecycleBot chat: {e}", exc_info=True)
            return ChatResponse(
                reply=f"⚠️ Encountered an error communicating with OpenAI: {str(e)[:150]}. Please verify your API key and quota.",
                model=self.default_model,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

gpt_service = GPTService()
