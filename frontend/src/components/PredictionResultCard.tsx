'use client';

import React, { useEffect } from 'react';
import confetti from 'canvas-confetti';
import {
  CheckCircle2,
  AlertTriangle,
  Download,
  Clock,
  Sparkles,
  Info,
  ChevronRight,
  ShieldCheck,
  Recycle
} from 'lucide-react';
import { PredictionResponse } from '@/lib/types';
import { CategoryBadge } from './CategoryBadge';
import { CATEGORY_META } from '@/lib/constants';
import { downloadFile } from '@/lib/utils';

interface PredictionResultCardProps {
  result: PredictionResponse;
  imageUrl?: string | null;
}

export const PredictionResultCard: React.FC<PredictionResultCardProps> = ({
  result,
  imageUrl,
}) => {
  const isRecyclable = result.guidance.is_recyclable;
  const meta = CATEGORY_META[result.prediction];

  // Trigger celebration confetti for recyclable items
  useEffect(() => {
    if (isRecyclable) {
      try {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.7 },
          colors: ['#10b981', '#059669', '#34d399', '#6ee7b7'],
        });
      } catch {
        // Confetti unavailable or blocked
      }
    }
  }, [result.id, isRecyclable]);

  const handleDownloadJson = () => {
    const payload = JSON.stringify(result, null, 2);
    downloadFile(payload, `recyclevision_prediction_${result.prediction}_${result.id.slice(0, 8)}.json`, 'application/json');
  };

  return (
    <div className="bg-white rounded-3xl border border-emerald-100 shadow-xl overflow-hidden transition-all animate-in fade-in-50 slide-in-from-bottom-4 duration-500">
      {/* Top Banner */}
      <div
        className={`px-6 sm:px-8 py-5 border-b flex flex-wrap items-center justify-between gap-3 ${
          isRecyclable
            ? 'bg-gradient-to-r from-emerald-600 to-green-700 text-white border-emerald-700'
            : 'bg-gradient-to-r from-slate-700 to-rose-700 text-white border-rose-800'
        }`}
      >
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-white/15 backdrop-blur-xs">
            {isRecyclable ? (
              <Recycle className="w-5 h-5 text-emerald-100" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-rose-100" />
            )}
          </div>
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-white/80 block">
              Classification Verified
            </span>
            <h3 className="text-xl font-bold tracking-tight">
              {isRecyclable ? 'Item is Recyclable' : 'General Landfill Trash'}
            </h3>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs bg-white/20 backdrop-blur-xs px-3 py-1 rounded-full font-mono text-white/90 inline-flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {result.inference_time_ms}ms
          </span>
          <button
            onClick={handleDownloadJson}
            className="text-xs bg-white text-slate-800 hover:bg-slate-100 font-semibold px-3 py-1.5 rounded-full shadow-xs transition-colors inline-flex items-center gap-1.5"
            title="Download prediction report as JSON"
          >
            <Download className="w-3.5 h-3.5" />
            Export JSON
          </button>
        </div>
      </div>

      <div className="p-6 sm:p-8 space-y-8">
        {/* Primary Prediction Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="space-y-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Primary Prediction
            </span>
            <div className="flex items-center gap-3">
              <CategoryBadge category={result.prediction} size="lg" showRecyclableBadge />
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">
              {result.guidance.summary}
            </p>
          </div>

          {/* Model Confidence Metric Card */}
          <div className="bg-slate-50/80 rounded-2xl p-5 border border-slate-200/80 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                Model Confidence
              </span>
              <span className="text-2xl font-black text-slate-900 font-mono">
                {result.confidence_percentage}%
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-1000 ease-out ${
                  result.confidence >= 0.85
                    ? 'bg-gradient-to-r from-emerald-500 to-green-600'
                    : result.confidence >= 0.65
                    ? 'bg-gradient-to-r from-teal-500 to-emerald-500'
                    : 'bg-gradient-to-r from-amber-500 to-rose-500'
                }`}
                style={{ width: `${result.confidence_percentage}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400">
              Model score produced via MobileNetV2 Softmax layer. Confidence represents model prediction probability.
            </p>
          </div>
        </div>

        {/* Top-3 Candidate Ranking */}
        <div className="space-y-3 pt-4 border-t border-slate-100">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            Top 3 Predicted Classes
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {result.top_predictions.map((item, idx) => (
              <div
                key={item.class_name}
                className={`p-3.5 rounded-xl border flex flex-col justify-between transition-all ${
                  idx === 0
                    ? 'bg-emerald-50/60 border-emerald-200 shadow-2xs'
                    : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold capitalize text-slate-700">
                    #{idx + 1} {item.class_name}
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-900">
                    {item.percentage}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-emerald-600 rounded-full"
                    style={{ width: `${item.percentage}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Actionable Recycling Guidelines */}
        <div className="space-y-4 pt-4 border-t border-slate-100">
          <h4 className="text-sm font-bold uppercase tracking-wider text-slate-900 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-600" />
            Actionable Recycling Guidance for {meta.label}
          </h4>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Disposal Guide */}
            <div className="bg-emerald-50/40 rounded-2xl p-5 border border-emerald-100 space-y-2">
              <h5 className="text-xs font-bold text-emerald-900 uppercase tracking-wide flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                How to Dispose
              </h5>
              <p className="text-xs text-slate-700 leading-relaxed">
                {result.guidance.disposal_guide}
              </p>
              <div className="pt-2">
                <span className="text-[11px] font-semibold text-emerald-800 block mb-1">
                  Common Examples:
                </span>
                <div className="flex flex-wrap gap-1">
                  {result.guidance.common_examples.map((ex) => (
                    <span
                      key={ex}
                      className="text-[10px] bg-white text-slate-700 px-2 py-0.5 rounded-md border border-emerald-200"
                    >
                      {ex}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Contamination Warnings */}
            <div className="bg-rose-50/40 rounded-2xl p-5 border border-rose-100 space-y-2">
              <h5 className="text-xs font-bold text-rose-900 uppercase tracking-wide flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-rose-600" />
                Contamination Warnings
              </h5>
              <ul className="space-y-1.5 text-xs text-slate-700">
                {result.guidance.contamination_warnings.map((warn, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-rose-500 font-bold">•</span>
                    <span>{warn}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Practical Tips */}
          <div className="bg-slate-50 rounded-2xl p-4 border border-slate-200 text-xs text-slate-600 space-y-1.5">
            <span className="font-semibold text-slate-800 block">
              💡 Best Practice Tips:
            </span>
            <ul className="space-y-1 pl-1">
              {result.guidance.recycling_tips.map((tip, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-emerald-600">✓</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
            <p className="text-[11px] text-slate-400 pt-2 italic border-t border-slate-200 mt-2">
              Note: {result.guidance.local_rules_note}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
