import os
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from backend.app.core.config import settings
from backend.app.schemas.prediction import PredictionResponse, PredictionRecord, PredictionItem
from backend.app.schemas.analytics import AnalyticsResponse, CategoryCount, DailyTrend

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.supabase_client = None
        self._init_supabase()
        self._init_sqlite()

    def _init_supabase(self):
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                from supabase import create_client, Client
                self.supabase_client: Client = create_client(
                    settings.SUPABASE_URL,
                    settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY
                )
                logger.info("Supabase client initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Supabase client: {e}. Falling back to SQLite.")
                self.supabase_client = None

    def _init_sqlite(self):
        db_path = settings.SQLITE_DB_PATH
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        predicted_class TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        top_predictions TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class);
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS activity_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        ip_address TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.commit()
                logger.info(f"SQLite database initialized at {db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")

    def save_prediction(self, pred: PredictionResponse, user_id: Optional[str] = None) -> bool:
        record_data = {
            "id": pred.id,
            "user_id": user_id,
            "predicted_class": pred.prediction,
            "confidence": pred.confidence,
            "top_predictions": json.dumps([item.model_dump() for item in pred.top_predictions]),
            "model_version": pred.model_version,
            "created_at": pred.timestamp
        }

        # 1. Try Supabase if configured
        if self.supabase_client:
            try:
                self.supabase_client.table("predictions").insert({
                    "id": pred.id,
                    "user_id": user_id,
                    "predicted_class": pred.prediction,
                    "confidence": pred.confidence,
                    "top_predictions": [item.model_dump() for item in pred.top_predictions],
                    "model_version": pred.model_version,
                    "created_at": pred.timestamp
                }).execute()
                logger.info(f"Saved prediction {pred.id} to Supabase.")
                return True
            except Exception as e:
                logger.warning(f"Failed to save prediction to Supabase: {e}. Falling back to SQLite.")

        # 2. Save to SQLite
        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (id, user_id, predicted_class, confidence, top_predictions, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_data["id"],
                    record_data["user_id"],
                    record_data["predicted_class"],
                    record_data["confidence"],
                    record_data["top_predictions"],
                    record_data["model_version"],
                    record_data["created_at"]
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save prediction to SQLite: {e}")
            return False

    def save_omni_prediction(self, pred: Any, user_id: Optional[str] = None) -> bool:
        """Saves a ChatGPT Omni-Prediction record into the database."""
        top_preds = [
            {"class_name": m.material, "score": (m.percentage or 50)/100.0, "percentage": float(m.percentage or 50)}
            for m in getattr(pred, "materials", [])
        ]
        record_data = {
            "id": pred.id,
            "user_id": user_id,
            "predicted_class": f"{pred.category}: {pred.item_name}"[:50],
            "confidence": pred.confidence_score,
            "top_predictions": json.dumps(top_preds),
            "model_version": pred.model_used,
            "created_at": pred.timestamp
        }

        # 1. Try Supabase
        if self.supabase_client:
            try:
                self.supabase_client.table("predictions").insert({
                    "id": pred.id,
                    "user_id": user_id,
                    "predicted_class": record_data["predicted_class"],
                    "confidence": pred.confidence_score,
                    "top_predictions": top_preds,
                    "model_version": pred.model_used,
                    "created_at": pred.timestamp
                }).execute()
                return True
            except Exception as e:
                logger.warning(f"Failed to save omni-prediction to Supabase: {e}. Falling back to SQLite.")

        # 2. SQLite
        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (id, user_id, predicted_class, confidence, top_predictions, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_data["id"],
                    record_data["user_id"],
                    record_data["predicted_class"],
                    record_data["confidence"],
                    record_data["top_predictions"],
                    record_data["model_version"],
                    record_data["created_at"]
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save omni-prediction to SQLite: {e}")
            return False


    def list_predictions(
        self,
        page: int = 1,
        limit: int = 20,
        category: Optional[str] = None,
        search: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[List[PredictionRecord], int]:
        offset = (page - 1) * limit
        query_conditions = []
        params = []

        if category:
            query_conditions.append("predicted_class = ?")
            params.append(category.lower())
        if user_id:
            query_conditions.append("user_id = ?")
            params.append(user_id)
        if start_date:
            query_conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            query_conditions.append("created_at <= ?")
            params.append(end_date)
        if search:
            query_conditions.append("(predicted_class LIKE ? OR id LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where_clause = ""
        if query_conditions:
            where_clause = "WHERE " + " AND ".join(query_conditions)

        records: List[PredictionRecord] = []
        total = 0

        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get total count
                count_query = f"SELECT COUNT(*) as count FROM predictions {where_clause}"
                cursor.execute(count_query, params)
                total = cursor.fetchone()["count"]

                # Fetch paginated records
                data_query = f"""
                    SELECT id, user_id, predicted_class, confidence, top_predictions, model_version, created_at
                    FROM predictions
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                cursor.execute(data_query, params + [limit, offset])
                rows = cursor.fetchall()

                for row in rows:
                    top_list = []
                    try:
                        raw_top = json.loads(row["top_predictions"])
                        for t in raw_top:
                            top_list.append(PredictionItem(**t))
                    except Exception:
                        pass

                    records.append(
                        PredictionRecord(
                            id=row["id"],
                            user_id=row["user_id"],
                            predicted_class=row["predicted_class"],
                            confidence=row["confidence"],
                            top_predictions=top_list,
                            model_version=row["model_version"],
                            created_at=row["created_at"]
                        )
                    )
        except Exception as e:
            logger.error(f"Error querying predictions: {e}")

        return records, total

    def get_analytics(self) -> AnalyticsResponse:
        total = 0
        cat_dist: List[CategoryCount] = []
        daily_trends: List[DailyTrend] = []
        avg_confidence = 0.0
        most_frequent = None
        recent_activity: List[Dict[str, Any]] = []

        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 1. Total & Avg Confidence
                cursor.execute("SELECT COUNT(*) as cnt, AVG(confidence) as avg_conf FROM predictions")
                row = cursor.fetchone()
                if row and row["cnt"]:
                    total = row["cnt"]
                    avg_confidence = round(float(row["avg_conf"] or 0.0), 4)

                if total > 0:
                    # 2. Category Distribution
                    cursor.execute("""
                        SELECT predicted_class, COUNT(*) as count
                        FROM predictions
                        GROUP BY predicted_class
                        ORDER BY count DESC
                    """)
                    dist_rows = cursor.fetchall()
                    for r in dist_rows:
                        pct = round((r["count"] / total) * 100.0, 2)
                        cat_dist.append(CategoryCount(
                            category=r["predicted_class"],
                            count=r["count"],
                            percentage=pct
                        ))
                    if cat_dist:
                        most_frequent = cat_dist[0].category

                    # 3. Daily trends (past 14 days)
                    cursor.execute("""
                        SELECT substr(created_at, 1, 10) as day, COUNT(*) as count
                        FROM predictions
                        GROUP BY day
                        ORDER BY day ASC
                        LIMIT 30
                    """)
                    trend_rows = cursor.fetchall()
                    for t in trend_rows:
                        daily_trends.append(DailyTrend(
                            date=t["day"],
                            count=t["count"]
                        ))

                    # 4. Recent activity
                    cursor.execute("""
                        SELECT id, predicted_class, confidence, created_at
                        FROM predictions
                        ORDER BY created_at DESC
                        LIMIT 10
                    """)
                    recent_rows = cursor.fetchall()
                    for r in recent_rows:
                        recent_activity.append({
                            "id": r["id"],
                            "category": r["predicted_class"],
                            "confidence": round(r["confidence"] * 100, 2),
                            "timestamp": r["created_at"]
                        })

        except Exception as e:
            logger.error(f"Error computing analytics: {e}")

        return AnalyticsResponse(
            total_predictions=total,
            category_distribution=cat_dist,
            most_frequent_category=most_frequent,
            daily_trends=daily_trends,
            average_confidence=avg_confidence,
            recent_activity=recent_activity
        )

    def log_activity(self, action: str, details: Optional[str] = None, user_id: Optional[str] = None, ip_address: Optional[str] = None):
        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO activity_logs (user_id, action, details, ip_address, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    action,
                    details,
                    ip_address,
                    datetime.utcnow().isoformat() + "Z"
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")

    def get_activity_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        logs = []
        try:
            with sqlite3.connect(settings.SQLITE_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, user_id, action, details, ip_address, created_at
                    FROM activity_logs
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                for row in cursor.fetchall():
                    logs.append(dict(row))
        except Exception as e:
            logger.error(f"Failed to fetch activity logs: {e}")
        return logs

db_service = DatabaseService()
