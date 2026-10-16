import {
  PredictionResponse,
  PredictionListResponse,
  PredictionRecord,
  AnalyticsResponse,
  ModelInfo,
  ModelMetricsResponse,
  AdminStats,
  ActivityLog,
  OmniPredictionResponse,
  ChatMessage,
  ChatResponse
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    if (!res.ok) {
      let errorMsg = `API request failed with status ${res.status}`;
      try {
        const errorData = await res.json();
        errorMsg = errorData.detail || errorData.message || errorMsg;
      } catch {
        // Body was not JSON
      }
      throw new ApiError(errorMsg, res.status);
    }

    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      err.message || 'Network error: could not connect to RecycleVision AI backend.',
      0
    );
  }
}

export const api = {
  // Health
  checkHealth: async () => {
    return request<{ status: string; service: string; version: string; model_loaded: boolean }>(
      '/health'
    );
  },

  // Model Metadata & Evaluation
  getModelInfo: async (): Promise<ModelInfo> => {
    return request<ModelInfo>('/api/v1/model/info');
  },

  getModelMetrics: async (): Promise<ModelMetricsResponse> => {
    return request<ModelMetricsResponse>('/api/v1/model/metrics');
  },

  // Predict
  predictWaste: async (file: File, token?: string): Promise<PredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const headers: Record<string, string> = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return request<PredictionResponse>('/api/v1/predict', {
      method: 'POST',
      body: formData,
      headers,
    });
  },

  // ChatGPT Vision Omni-Prediction (Predict ANY item or material)
  predictOmniWaste: async (file: File, apiKey?: string, token?: string): Promise<OmniPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const headers: Record<string, string> = {};
    if (apiKey) {
      headers['X-OpenAI-Key'] = apiKey;
    }
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    return request<OmniPredictionResponse>('/api/v1/predict/omni', {
      method: 'POST',
      body: formData,
      headers,
    });
  },

  // RecycleBot AI Chat
  chatWithRecycleBot: async (messages: ChatMessage[], apiKey?: string): Promise<ChatResponse> => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) {
      headers['X-OpenAI-Key'] = apiKey;
    }

    return request<ChatResponse>('/api/v1/chat', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        messages,
        user_api_key: apiKey || null,
      }),
    });
  },


  // Analytics
  getAnalytics: async (): Promise<AnalyticsResponse> => {
    return request<AnalyticsResponse>('/api/v1/analytics');
  },

  // Prediction History
  getPredictions: async (params: {
    page?: number;
    limit?: number;
    category?: string;
    search?: string;
    startDate?: string;
    endDate?: string;
    token?: string;
  } = {}): Promise<PredictionListResponse> => {
    const q = new URLSearchParams();
    if (params.page) q.append('page', params.page.toString());
    if (params.limit) q.append('limit', params.limit.toString());
    if (params.category && params.category !== 'all') q.append('category', params.category);
    if (params.search) q.append('search', params.search);
    if (params.startDate) q.append('start_date', params.startDate);
    if (params.endDate) q.append('end_date', params.endDate);

    const headers: Record<string, string> = {};
    if (params.token) {
      headers['Authorization'] = `Bearer ${params.token}`;
    }

    return request<PredictionListResponse>(`/api/v1/predictions?${q.toString()}`, {
      headers,
    });
  },

  getPredictionDetail: async (id: string): Promise<PredictionRecord> => {
    return request<PredictionRecord>(`/api/v1/predictions/${id}`);
  },

  getExportCsvUrl: (category?: string): string => {
    const q = category && category !== 'all' ? `?category=${category}` : '';
    return `${API_BASE_URL}/api/v1/predictions/export/csv${q}`;
  },

  // Admin Endpoints
  getAdminStats: async (adminKey: string): Promise<AdminStats> => {
    return request<AdminStats>('/api/v1/admin/stats', {
      headers: {
        'X-Admin-Key': adminKey,
      },
    });
  },

  getAdminLogs: async (adminKey: string, limit: number = 50): Promise<ActivityLog[]> => {
    return request<ActivityLog[]>(`/api/v1/admin/logs?limit=${limit}`, {
      headers: {
        'X-Admin-Key': adminKey,
      },
    });
  },

  seedDemoData: async (adminKey: string) => {
    return request<{ status: string; records_seeded: number; message: string }>(
      '/api/v1/admin/seed-demo',
      {
        method: 'POST',
        headers: {
          'X-Admin-Key': adminKey,
        },
      }
    );
  },
};
