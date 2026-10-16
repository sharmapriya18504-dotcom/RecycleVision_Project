export type WasteClass = 'cardboard' | 'glass' | 'metal' | 'paper' | 'plastic' | 'trash';

export interface PredictionItem {
  class_name: WasteClass;
  score: number;
  percentage: number;
}

export interface RecyclingGuidance {
  category: WasteClass;
  is_recyclable: boolean;
  summary: string;
  common_examples: string[];
  disposal_guide: string;
  recycling_tips: string[];
  contamination_warnings: string[];
  local_rules_note?: string;
}

export interface PredictionResponse {
  id: string;
  prediction: WasteClass;
  confidence: number;
  confidence_percentage: number;
  top_predictions: PredictionItem[];
  guidance: RecyclingGuidance;
  model_version: string;
  inference_time_ms: number;
  timestamp: string;
}

export interface PredictionRecord {
  id: string;
  user_id?: string | null;
  predicted_class: WasteClass;
  confidence: number;
  top_predictions: PredictionItem[];
  model_version: string;
  created_at: string;
}

export interface PredictionListResponse {
  items: PredictionRecord[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface CategoryCount {
  category: WasteClass;
  count: number;
  percentage: number;
}

export interface DailyTrend {
  date: string;
  count: number;
}

export interface AnalyticsResponse {
  total_predictions: number;
  category_distribution: CategoryCount[];
  most_frequent_category: WasteClass | null;
  daily_trends: DailyTrend[];
  average_confidence: number;
  recent_activity: {
    id: string;
    category: WasteClass;
    confidence: number;
    timestamp: string;
  }[];
}

export interface ModelInfo {
  model_name: string;
  architecture: string;
  base_model: string;
  input_shape: (number | null)[];
  classes: WasteClass[];
  num_classes: number;
  total_parameters: number;
  trainable_parameters: number;
  non_trainable_parameters: number;
  normalization: string;
  activation: string;
  model_version: string;
  status: string;
}

export interface ClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ConfusionMatrixData {
  classes: WasteClass[];
  matrix: number[][];
}

export interface TrainingEpochMetric {
  epoch: number;
  train_accuracy: number;
  val_accuracy: number;
  train_loss: number;
  val_loss: number;
}

export interface ModelMetricsResponse {
  status: string;
  test_accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_f1: number;
  evaluated_samples: number;
  class_metrics: Record<WasteClass, ClassMetric>;
  confusion_matrix: ConfusionMatrixData;
  training_history: TrainingEpochMetric[];
  evaluation_dataset: string;
  evaluated_at: string;
  known_limitations: string[];
}

export interface AdminStats {
  system_status: string;
  timestamp: string;
  authenticated_admin: string;
  auth_type: string;
  model: {
    loaded: boolean;
    architecture: string;
    classes: string[];
    parameters: number;
    version: string;
  };
  database: {
    mode: string;
    total_records: number;
    storage_path?: string;
  };
  resources: {
    rss_memory_mb: number;
    vms_memory_mb: number;
    cpu_percent: number;
  };
}

export interface ActivityLog {
  id: number;
  user_id?: string | null;
  action: string;
  details?: string | null;
  ip_address?: string | null;
  created_at: string;
}

// --- ChatGPT Omni-Prediction & Chat Interfaces ---

export interface OmniMaterialItem {
  material: string;
  percentage?: number | null;
  is_recyclable: boolean;
}

export interface OmniPredictionResponse {
  id: string;
  engine: string;
  item_name: string;
  category: string;
  is_recyclable: boolean;
  confidence_score: number;
  confidence_percentage: number;
  summary: string;
  materials: OmniMaterialItem[];
  disposal_instructions: string[];
  special_warnings: string[];
  environmental_impact: string;
  upcycling_ideas: string[];
  model_used: string;
  inference_time_ms: number;
  timestamp: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  reply: string;
  model: string;
  timestamp: string;
}

