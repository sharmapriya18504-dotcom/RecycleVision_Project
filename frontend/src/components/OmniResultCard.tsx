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
  ShieldAlert,
  Leaf,
  Layers,
  ArrowRight,
  Lightbulb,
  Cpu
} from 'lucide-react';
import { OmniPredictionResponse } from '@/lib/types';
import { downloadFile } from '@/lib/utils';

interface OmniResultCardProps {
  result: OmniPredictionResponse;
}

export const OmniResultCard: React.FC<OmniResultCardProps> = ({ result }) => {
  const isRecyclable = result.is_recyclable;

  useEffect(() => {
    if (isRecyclable) {
      try {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.7 },
          colors: ['#8b5cf6', '#6366f1', '#10b981', '#34d399'],
        });
      } catch {
        // Confetti unavailable
      }
    }
  }, [result.id, isRecyclable]);

  const handleDownloadJson = () => {
    const payload = JSON.stringify(result, null, 2);
    downloadFile(payload, `omni_prediction_${result.item_name.replace(/\s+/g, '_')}_${result.id}.json`, 'application/json');
  };

  return (
    <div className="bg-white rounded-3xl border border-purple-200 shadow-xl overflow-hidden transition-all animate-in fade-in-50 slide-in-from-bottom-4 duration-500">
      {/* Top Banner */}
      <div className="px-6 sm:px-8 py-5 border-b flex flex-wrap items-center justify-between gap-3 bg-gradient-to-r from-purple-700 via-indigo-700 to-slate-900 text-white border-purple-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-white/10 backdrop-blur-md">
            <Sparkles className="w-6 h-6 text-purple-200" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-purple-200 bg-white/10 px-2 py-0.5 rounded-full">
                ChatGPT Vision Omni-Mode
              </span>
              <span className="text-[11px] font-mono text-purple-200">
                {result.model_used}
              </span>
            </div>
            <h3 className="text-xl sm:text-2xl font-black tracking-tight capitalize">
              {result.item_name}
            </h3>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Recyclability status badge */}
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
              isRecyclable
                ? 'bg-emerald-500 text-white'
                : 'bg-rose-500 text-white'
            }`}
          >
            {isRecyclable ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Recyclable Stream</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Specialized / Non-Recyclable</span>
              </>
            )}
          </span>

          {/* Export JSON */}
          <button
            onClick={handleDownloadJson}
            className="p-2 rounded-xl bg-white/10 hover:bg-white/20 text-white transition-all text-xs font-semibold inline-flex items-center gap-1"
            title="Download JSON telemetry payload"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Content Body */}
      <div className="p-6 sm:p-8 space-y-6">
        {/* Core Stats Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 space-y-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Identified Category
            </span>
            <p className="text-lg font-bold text-slate-900 capitalize">
              {result.category}
            </p>
          </div>

          <div className="bg-purple-50/60 p-4 rounded-2xl border border-purple-100 space-y-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-purple-700">
              AI Confidence Score
            </span>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-black text-purple-900">
                {result.confidence_percentage}%
              </span>
              <span className="text-xs text-purple-600 font-medium">high certainty</span>
            </div>
          </div>

          <div className="bg-slate-50 p-4 rounded-2xl border border-slate-100 space-y-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Inference Latency
            </span>
            <div className="flex items-center gap-1.5 text-slate-900 font-mono font-bold text-lg">
              <Clock className="w-4 h-4 text-slate-400" />
              <span>{result.inference_time_ms} ms</span>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100 text-slate-700 text-sm leading-relaxed">
          <div className="flex items-center gap-2 text-indigo-900 font-semibold mb-1 text-xs">
            <Info className="w-4 h-4 text-indigo-600" />
            <span>AI Assessment Overview</span>
          </div>
          {result.summary}
        </div>

        {/* Material Composition Breakdown */}
        {result.materials && result.materials.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-600" />
              <span>Detected Material Composition</span>
            </h4>

            <div className="space-y-2.5">
              {result.materials.map((mat, i) => (
                <div key={i} className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 space-y-1.5">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-slate-800">{mat.material}</span>
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          mat.is_recyclable
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-slate-200 text-slate-700'
                        }`}
                      >
                        {mat.is_recyclable ? 'Recyclable' : 'Non-Recyclable'}
                      </span>
                      {mat.percentage && (
                        <span className="text-slate-500 font-mono">~{mat.percentage}%</span>
                      )}
                    </div>
                  </div>
                  {mat.percentage && (
                    <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          mat.is_recyclable ? 'bg-emerald-500' : 'bg-slate-400'
                        }`}
                        style={{ width: `${mat.percentage}%` }}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Critical Safety & Contamination Warnings */}
        {result.special_warnings && result.special_warnings.length > 0 && (
          <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 space-y-2">
            <div className="flex items-center gap-2 text-amber-900 font-bold text-xs uppercase tracking-wider">
              <ShieldAlert className="w-4 h-4 text-amber-700" />
              <span>Safety & Contamination Warnings</span>
            </div>
            <ul className="space-y-1 text-xs text-amber-800">
              {result.special_warnings.map((warn, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-amber-600 font-bold">•</span>
                  <span>{warn}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Step-by-Step Disposal Instructions */}
        <div className="space-y-3">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            Recommended Actionable Disposal Steps
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {result.disposal_instructions.map((step, idx) => (
              <div
                key={idx}
                className="bg-slate-50/80 p-3.5 rounded-xl border border-slate-200/80 flex items-start gap-3"
              >
                <div className="w-6 h-6 rounded-full bg-purple-600 text-white font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                  {idx + 1}
                </div>
                <p className="text-xs text-slate-700 leading-snug">{step}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Environmental Impact & Upcycling Ideas */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          {/* Eco Impact */}
          <div className="p-4 rounded-2xl bg-emerald-50/60 border border-emerald-100 space-y-1.5">
            <div className="flex items-center gap-2 text-emerald-900 font-bold text-xs">
              <Leaf className="w-4 h-4 text-emerald-600" />
              <span>Environmental & Carbon Impact</span>
            </div>
            <p className="text-xs text-emerald-800 leading-relaxed">
              {result.environmental_impact}
            </p>
          </div>

          {/* Upcycling Ideas */}
          {result.upcycling_ideas && result.upcycling_ideas.length > 0 && (
            <div className="p-4 rounded-2xl bg-amber-50/50 border border-amber-100 space-y-1.5">
              <div className="flex items-center gap-2 text-amber-900 font-bold text-xs">
                <Lightbulb className="w-4 h-4 text-amber-600" />
                <span>Upcycling & Creative Reuse Ideas</span>
              </div>
              <ul className="text-xs text-amber-800 space-y-1">
                {result.upcycling_ideas.map((idea, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-amber-500">•</span>
                    <span>{idea}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
