'use client';

import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  Award,
  Calendar,
  Layers,
  Sparkles,
  RefreshCw,
  AlertCircle,
  Filter,
  CheckCircle2,
  Clock
} from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';
import { CategoryDistributionChart } from '@/components/CategoryDistributionChart';
import { DailyTrendsChart } from '@/components/DailyTrendsChart';
import { CategoryBadge } from '@/components/CategoryBadge';
import { api } from '@/lib/api';
import { AnalyticsResponse, WasteClass } from '@/lib/types';
import { formatDate } from '@/lib/utils';
import { WASTE_CLASSES } from '@/lib/constants';

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const fetchAnalytics = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await api.getAnalytics();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch analytics from the backend.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  // Filter category distribution if a category is filtered
  const filteredDistribution = React.useMemo(() => {
    if (!data) return [];
    if (selectedCategory === 'all') return data.category_distribution;
    return data.category_distribution.filter((c) => c.category === selectedCategory);
  }, [data, selectedCategory]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 mb-2">
            <BarChart3 className="w-3.5 h-3.5" />
            Live Environmental Intelligence
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Analytics Dashboard
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Real-time classification analytics computed directly from stored prediction records.
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          disabled={isLoading}
          className="inline-flex items-center gap-1.5 text-xs font-semibold bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 px-4 py-2 rounded-xl shadow-2xs transition-colors self-start sm:self-auto cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh Data
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 flex items-center gap-3 text-xs text-rose-800">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Metric Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="Total Predictions"
          value={data?.total_predictions ?? (isLoading ? '...' : 0)}
          subtitle="Processed waste items"
          icon={Layers}
          accentColor="emerald"
        />
        <MetricCard
          title="Most Frequent Class"
          value={data?.most_frequent_category ? data.most_frequent_category.toUpperCase() : 'N/A'}
          subtitle="Highest disposal volume"
          icon={TrendingUp}
          accentColor="teal"
        />
        <MetricCard
          title="Average Confidence"
          value={
            data?.average_confidence
              ? `${(data.average_confidence * 100).toFixed(1)}%`
              : (isLoading ? '...' : '0.0%')
          }
          subtitle="MobileNetV2 certainty"
          icon={Sparkles}
          accentColor="blue"
        />
        <MetricCard
          title="Model Test Accuracy"
          value="78.53%"
          subtitle="Verified on 503 test samples"
          icon={Award}
          accentColor="amber"
        />
      </div>

      {/* Filter Control Bar */}
      <div className="bg-white p-4 rounded-2xl border border-slate-200 flex flex-wrap items-center justify-between gap-4 shadow-2xs">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-700">
          <Filter className="w-4 h-4 text-emerald-600" />
          <span>Filter Distribution:</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-colors ${
              selectedCategory === 'all'
                ? 'bg-emerald-700 text-white shadow-2xs'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            All Categories
          </button>
          {WASTE_CLASSES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`text-xs px-3 py-1.5 rounded-lg font-medium capitalize transition-colors ${
                selectedCategory === cat
                  ? 'bg-emerald-700 text-white shadow-2xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Recharts Visualizations */}
      {data && (
        <div className="space-y-8">
          {/* Category Distribution (Bar & Pie) */}
          <CategoryDistributionChart data={filteredDistribution} />

          {/* Daily Trends Area Chart */}
          <DailyTrendsChart data={data.daily_trends} />
        </div>
      )}

      {/* Recent Activity Table */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Clock className="w-5 h-5 text-emerald-600" />
              Recent Prediction Activity
            </h3>
            <p className="text-xs text-slate-500">
              Latest 10 real-time classifications logged to the database
            </p>
          </div>
        </div>

        {data?.recent_activity && data.recent_activity.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3 px-3">Prediction ID</th>
                  <th className="py-3 px-3">Class</th>
                  <th className="py-3 px-3">Confidence</th>
                  <th className="py-3 px-3">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {data.recent_activity.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3 px-3 font-mono text-slate-500">
                      {item.id.slice(0, 12)}...
                    </td>
                    <td className="py-3 px-3">
                      <CategoryBadge category={item.category} size="sm" />
                    </td>
                    <td className="py-3 px-3 font-mono font-bold text-slate-800">
                      {item.confidence.toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-slate-500">
                      {formatDate(item.timestamp)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-10 text-slate-400 text-xs">
            No activity records logged yet. Head over to the Classify page to test an image!
          </div>
        )}
      </div>
    </div>
  );
}
