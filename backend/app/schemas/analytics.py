from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class CategoryCount(BaseModel):
    category: str
    count: int
    percentage: float

class DailyTrend(BaseModel):
    date: str
    count: int

class AnalyticsResponse(BaseModel):
    total_predictions: int
    category_distribution: List[CategoryCount]
    most_frequent_category: Optional[str] = None
    daily_trends: List[DailyTrend]
    average_confidence: float
    recent_activity: List[Dict[str, Any]]
