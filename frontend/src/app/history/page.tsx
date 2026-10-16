'use client';

import React, { useState, useEffect } from 'react';
import {
  History,
  Search,
  Filter,
  Download,
  RefreshCw,
  AlertCircle,
  Eye,
  X,
  Clock,
  Sparkles,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { CategoryBadge } from '@/components/CategoryBadge';
import { api } from '@/lib/api';
import { PredictionRecord, PredictionListResponse, WasteClass } from '@/lib/types';
import { formatDate } from '@/lib/utils';
import { WASTE_CLASSES } from '@/lib/constants';

export default function HistoryPage() {
  const [data, setData] = useState<PredictionListResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const [selectedRecord, setSelectedRecord] = useState<PredictionRecord | null>(null);

  const fetchHistory = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await api.getPredictions({
        page,
        limit: 15,
        category: selectedCategory,
        search: searchQuery.trim() || undefined,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch prediction history from backend.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, selectedCategory]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchHistory();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 mb-2">
            <History className="w-3.5 h-3.5" />
            Audit & Inference Logs
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Prediction History
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Browse, filter, and inspect past waste classification events stored in the database.
          </p>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <a
            href={api.getExportCsvUrl(selectedCategory)}
            download="recyclevision_predictions.csv"
            className="inline-flex items-center gap-1.5 text-xs font-semibold bg-emerald-700 hover:bg-emerald-800 text-white px-4 py-2.5 rounded-xl shadow-xs transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </a>
          <button
            onClick={fetchHistory}
            disabled={isLoading}
            className="p-2.5 rounded-xl border border-slate-300 text-slate-600 hover:bg-slate-50 transition-colors"
            title="Refresh history"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white p-4 sm:p-5 rounded-3xl border border-slate-200 shadow-xs flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search */}
        <form onSubmit={handleSearchSubmit} className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by ID or class..."
            className="w-full pl-10 pr-4 py-2 text-xs rounded-xl border border-slate-200 focus:outline-hidden focus:border-emerald-500 bg-slate-50/50"
          />
        </form>

        {/* Category Pills */}
        <div className="flex items-center gap-1.5 flex-wrap w-full md:w-auto">
          <button
            onClick={() => {
              setSelectedCategory('all');
              setPage(1);
            }}
            className={`text-xs px-3 py-1.5 rounded-xl font-medium transition-colors ${
              selectedCategory === 'all'
                ? 'bg-emerald-700 text-white shadow-2xs'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            All
          </button>
          {WASTE_CLASSES.map((cat) => (
            <button
              key={cat}
              onClick={() => {
                setSelectedCategory(cat);
                setPage(1);
              }}
              className={`text-xs px-3 py-1.5 rounded-xl font-medium capitalize transition-colors ${
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

      {/* Error state */}
      {error && (
        <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 flex items-center gap-3 text-xs text-rose-800">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* History Data Table */}
      <div className="bg-white rounded-3xl border border-slate-200 shadow-xs overflow-hidden">
        {data && data.items.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3.5 px-4">Prediction ID</th>
                  <th className="py-3.5 px-4">Class</th>
                  <th className="py-3.5 px-4">Confidence</th>
                  <th className="py-3.5 px-4">Top 3 Ranking</th>
                  <th className="py-3.5 px-4">Model Version</th>
                  <th className="py-3.5 px-4">Timestamp</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {data.items.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => setSelectedRecord(item)}
                    className="hover:bg-emerald-50/40 transition-colors cursor-pointer"
                  >
                    <td className="py-3.5 px-4 font-mono text-slate-600">
                      {item.id.slice(0, 10)}...
                    </td>
                    <td className="py-3.5 px-4">
                      <CategoryBadge category={item.predicted_class} size="sm" showRecyclableBadge />
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-slate-800">
                          {(item.confidence * 100).toFixed(1)}%
                        </span>
                        <div className="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden hidden sm:block">
                          <div
                            className="bg-emerald-600 h-full rounded-full"
                            style={{ width: `${item.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-slate-500">
                      <span className="capitalize">
                        {item.top_predictions.map((p) => p.class_name).join(' • ')}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-400">
                      v{item.model_version}
                    </td>
                    <td className="py-3.5 px-4 text-slate-500">
                      {formatDate(item.created_at)}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRecord(item);
                        }}
                        className="text-xs text-emerald-700 hover:text-emerald-900 font-semibold inline-flex items-center gap-1"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-16 text-center text-slate-400 text-xs space-y-2">
            <History className="w-8 h-8 text-slate-300 mx-auto" />
            <p className="font-medium text-slate-600 text-sm">No predictions found.</p>
            <p>Upload a waste image in the Classify page to generate inference records.</p>
          </div>
        )}

        {/* Pagination Bar */}
        {data && data.pages > 1 && (
          <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <div>
              Showing page <span className="font-bold text-slate-800">{data.page}</span> of{' '}
              <span className="font-bold text-slate-800">{data.pages}</span> ({data.total} total)
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={data.page <= 1}
                className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
                disabled={data.page >= data.pages}
                className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Record Inspection Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl max-w-lg w-full p-6 sm:p-7 space-y-5 border border-slate-200 shadow-2xl relative animate-in fade-in zoom-in-95 duration-200">
            <button
              onClick={() => setSelectedRecord(null)}
              className="absolute top-4 right-4 p-1.5 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="space-y-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                Prediction Record #{selectedRecord.id}
              </span>
              <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                <CategoryBadge category={selectedRecord.predicted_class} size="lg" showRecyclableBadge />
              </h3>
            </div>

            <div className="space-y-3 bg-slate-50 p-4 rounded-2xl border border-slate-200 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Confidence Score:</span>
                <span className="font-mono font-bold text-slate-900">
                  {(selectedRecord.confidence * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Model Version:</span>
                <span className="font-mono text-slate-900">v{selectedRecord.model_version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Recorded At:</span>
                <span className="text-slate-900">{formatDate(selectedRecord.created_at)}</span>
              </div>
            </div>

            {/* Top 3 Breakdown */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Candidate Classes
              </h4>
              <div className="space-y-1.5">
                {selectedRecord.top_predictions.map((p, idx) => (
                  <div
                    key={p.class_name}
                    className="flex items-center justify-between text-xs p-2.5 rounded-xl bg-slate-50 border border-slate-200/80"
                  >
                    <span className="font-semibold capitalize">
                      #{idx + 1} {p.class_name}
                    </span>
                    <span className="font-mono font-bold text-emerald-700">
                      {p.percentage}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedRecord(null)}
                className="text-xs bg-slate-900 hover:bg-slate-800 text-white font-semibold px-5 py-2.5 rounded-xl transition-colors"
              >
                Close Record
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
