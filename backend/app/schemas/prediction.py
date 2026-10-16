from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Waste category name")
    score: float = Field(..., description="Softmax confidence score (0.0 to 1.0)")
    percentage: float = Field(..., description="Percentage formatted confidence (0 to 100%)")

class RecyclingGuidance(BaseModel):
    category: str
    is_recyclable: bool
    summary: str
    common_examples: List[str]
    disposal_guide: str
    recycling_tips: List[str]
    contamination_warnings: List[str]
    local_rules_note: str = "Recycling capabilities vary significantly by municipality. Always check your local waste management guidelines."

class PredictionResponse(BaseModel):
    id: str = Field(..., description="Unique ID for this prediction event")
    prediction: str = Field(..., description="Primary predicted waste category")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    confidence_percentage: float = Field(..., description="Confidence represented as percentage")
    top_predictions: List[PredictionItem] = Field(..., description="Top 3 ranked predictions")
    guidance: RecyclingGuidance = Field(..., description="Practical recycling guidance for the predicted item")
    model_version: str = Field(..., description="Configured model version")
    inference_time_ms: float = Field(..., description="Time taken to process inference in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")

class PredictionRecord(BaseModel):
    id: str
    user_id: Optional[str] = None
    predicted_class: str
    confidence: float
    top_predictions: List[PredictionItem]
    model_version: str
    created_at: str

class PredictionListResponse(BaseModel):
    items: List[PredictionRecord]
    total: int
    page: int
    limit: int
    pages: int

# --- ChatGPT Omni-Prediction & Chat Schemas ---

class OmniMaterialItem(BaseModel):
    material: str = Field(..., description="Identified material component")
    percentage: Optional[int] = Field(None, description="Estimated percentage share")
    is_recyclable: bool = Field(..., description="Recyclability of this specific component")

class OmniPredictionResponse(BaseModel):
    id: str = Field(..., description="Unique prediction ID")
    engine: str = Field("chatgpt-vision", description="Prediction engine used")
    item_name: str = Field(..., description="Specific item identified (e.g., Lithium-Ion Smartphone Battery)")
    category: str = Field(..., description="Broad category: e-waste, hazardous, plastic, textile, organic, etc.")
    is_recyclable: bool = Field(..., description="Whether the item is generally recyclable")
    confidence_score: float = Field(..., description="Confidence between 0.0 and 1.0")
    confidence_percentage: float = Field(..., description="Confidence represented as percentage")
    summary: str = Field(..., description="Brief summary of item composition and status")
    materials: List[OmniMaterialItem] = Field(default=[], description="Detected material breakdown")
    disposal_instructions: List[str] = Field(..., description="Step-by-step proper disposal instructions")
    special_warnings: List[str] = Field(default=[], description="Safety, contamination, or fire hazard warnings")
    environmental_impact: str = Field(..., description="Eco-impact of proper vs improper disposal")
    upcycling_ideas: List[str] = Field(default=[], description="Creative repurposing or reuse ideas")
    model_used: str = Field(..., description="AI model used (e.g. gpt-4o-mini)")
    inference_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message text content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history messages")
    user_api_key: Optional[str] = Field(None, description="Optional client-provided OpenAI API key")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Assistant reply message")
    model: str = Field(..., description="AI model used")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

