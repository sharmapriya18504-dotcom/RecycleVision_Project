import logging
from typing import Optional
from fastapi import APIRouter, Header, Depends, Request

from backend.app.core.security import get_current_user_optional
from backend.app.schemas.prediction import ChatRequest, ChatResponse
from backend.app.services.gpt_service import gpt_service
from backend.app.services.database import db_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("", response_model=ChatResponse, summary="Chat with RecycleBot AI")
async def chat_with_recyclebot(
    chat_req: ChatRequest,
    request: Request,
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key", description="Optional client OpenAI API Key"),
    current_user: dict = Depends(get_current_user_optional)
):
    """
    Conversational recycling and sustainability assistant powered by ChatGPT.
    Answers queries on safe disposal, e-waste, hazardous materials, and municipal sorting.
    Supports English, Hindi, and Hinglish.
    """
    effective_key = x_openai_key or chat_req.user_api_key
    response = gpt_service.chat_recyclebot(chat_req.messages, user_api_key=effective_key)

    user_id = current_user.get("id") if current_user else None
    client_ip = request.client.host if request.client else "unknown"
    db_service.log_activity(
        action="CHAT_COMPLETION",
        details="Interacted with RecycleBot AI",
        user_id=user_id,
        ip_address=client_ip
    )

    return response
