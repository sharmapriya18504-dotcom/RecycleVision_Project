'use client';

import React from 'react';
import Link from 'next/link';
import {
  ScanLine,
  BarChart3,
  BookOpen,
  Award,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  Clock,
  ShieldCheck,
  Cpu,
  Layers,
  HelpCircle,
  TrendingUp,
  Recycle
} from 'lucide-react';
import { CATEGORY_META, WASTE_CLASSES, CREATOR_INFO } from '@/lib/constants';
import { CategoryBadge } from '@/components/CategoryBadge';

export default function HomePage() {
  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-emerald-800 via-green-800 to-teal-900 text-white p-6 sm:p-10 shadow-lg border border-emerald-700/50">
        <div className="relative z-10 max-w-2xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/15 backdrop-blur-md text-emerald-200 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-amber-300" />
            <span>MobileNetV2 Transfer Learning • 78.53% Verified Accuracy</span>
          </div>

          <h1 className="text-2xl sm:text-4xl lg:text-5xl font-black tracking-tight leading-tight">
            Smart Waste Classification <br />
            <span className="text-emerald-300">&amp; Recycling Analytics</span>
          </h1>

          <p className="text-xs sm:text-sm text-emerald-100/90 leading-relaxed max-w-xl">
            Upload any waste photograph. Our deep learning model instantly predicts the material category, evaluates confidence, and gives you actionable disposal instructions to prevent batch contamination.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <Link
              href="/classify"
              className="inline-flex items-center gap-2 bg-white text-emerald-900 hover:bg-emerald-50 px-5 py-3 rounded-2xl text-xs sm:text-sm font-bold shadow-md hover:shadow-lg transition-all active:scale-98"
            >
              <ScanLine className="w-4 h-4 text-emerald-700" />
              <span>Start Waste Classification</span>
              <ArrowRight className="w-3.5 h-3.5 text-emerald-700" />
            </Link>

            <Link
              href="/analytics"
              className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white border border-white/20 px-5 py-3 rounded-2xl text-xs sm:text-sm font-semibold transition-all active:scale-98"
            >
              <BarChart3 className="w-4 h-4 text-emerald-300" />
              <span>Explore Analytics</span>
            </Link>
          </div>
        </div>

        {/* Decorative background element */}
        <div className="absolute right-[-40px] bottom-[-40px] opacity-10 pointer-events-none hidden md:block">
          <Recycle className="w-96 h-96 text-white" />
        </div>
      </div>

      {/* 4 Quick Stat Tiles */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Test Accuracy
            </span>
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-700">
              <Award className="w-4 h-4" />
            </div>
          </div>
          <div className="text-xl sm:text-2xl font-black text-slate-900 font-mono">
            78.53%
          </div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">
            Verified on 503 test images
          </span>
        </div>

        <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Categories
            </span>
            <div className="p-2 rounded-xl bg-teal-50 text-teal-700">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="text-xl sm:text-2xl font-black text-slate-900 font-mono">
            6 Classes
          </div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">
            Cardboard, Glass, Metal...
          </span>
        </div>

        <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Speed
            </span>
            <div className="p-2 rounded-xl bg-sky-50 text-sky-700">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="text-xl sm:text-2xl font-black text-slate-900 font-mono">
            ~30ms
          </div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">
            Fast CPU inference latency
          </span>
        </div>

        <div className="bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-2xs hover:shadow-xs transition-shadow">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              AI Backbone
            </span>
            <div className="p-2 rounded-xl bg-amber-50 text-amber-700">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <div className="text-xl sm:text-2xl font-black text-slate-900 font-mono">
            MobileNetV2
          </div>
          <span className="text-[11px] text-slate-500 mt-0.5 block">
            2.42M parameters
          </span>
        </div>
      </div>

      {/* 3-Step Simple Guide */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-slate-900 tracking-tight">
              How It Works — Simple &amp; Automated
            </h3>
            <p className="text-xs text-slate-500">
              Classifying garbage and recycling correctly takes 3 easy steps.
            </p>
          </div>
          <Link
            href="/classify"
            className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 hidden sm:inline-flex items-center gap-1"
          >
            Try it now →
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="bg-slate-50/70 p-5 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-600 text-white flex items-center justify-center text-xs font-black">
              1
            </div>
            <h4 className="text-sm font-bold text-slate-900">Upload Your Photo</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Drag &amp; drop an image of a waste item or pick one from your files or curated samples.
            </p>
          </div>

          <div className="bg-slate-50/70 p-5 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-600 text-white flex items-center justify-center text-xs font-black">
              2
            </div>
            <h4 className="text-sm font-bold text-slate-900">AI Analyzes Features</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              MobileNetV2 detects visual textures, shapes, and materials through 157 neural layers.
            </p>
          </div>

          <div className="bg-slate-50/70 p-5 rounded-2xl border border-slate-200/80 space-y-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-600 text-white flex items-center justify-center text-xs font-black">
              3
            </div>
            <h4 className="text-sm font-bold text-slate-900">Get Clear Disposal Advice</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Check if the item is recyclable, review clean-up tips, and prevent contamination.
            </p>
          </div>
        </div>
      </div>

      {/* 6 Supported Categories Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-slate-900 tracking-tight">
              Supported Waste Categories
            </h3>
            <p className="text-xs text-slate-500">
              Click any category card to view full recycling and preparation guidelines.
            </p>
          </div>
          <Link
            href="/guide"
            className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 inline-flex items-center gap-1"
          >
            Full Guide →
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {WASTE_CLASSES.map((category) => {
            const meta = CATEGORY_META[category];
            const isRecyclable = category !== 'trash';

            return (
              <Link
                key={category}
                href="/guide"
                className="bg-white rounded-2xl p-5 border border-slate-200 hover:border-emerald-300 shadow-2xs hover:shadow-xs transition-all group flex flex-col justify-between space-y-3"
              >
                <div className="flex items-center justify-between">
                  <CategoryBadge category={category} size="sm" />
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                      isRecyclable
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : 'bg-rose-50 text-rose-700 border border-rose-200'
                    }`}
                  >
                    {isRecyclable ? 'Recyclable' : 'Landfill'}
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {meta.description}
                </p>
                <div className="text-[11px] font-semibold text-emerald-700 group-hover:text-emerald-800 flex items-center gap-1">
                  <span>View disposal rules</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Creator Credits Footer Card */}
      <div className="bg-slate-100/70 rounded-2xl p-4 sm:p-5 border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-600">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-xs">
            PS
          </div>
          <div>
            <span className="font-bold text-slate-900 block">
              {CREATOR_INFO.name} • {CREATOR_INFO.degree}
            </span>
            <span className="text-[11px] text-slate-500">
              Academic Computer Science Portfolio Project
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/about"
            className="text-emerald-700 hover:text-emerald-800 font-semibold underline"
          >
            Viva Notes &amp; Documentation
          </Link>
          <span>•</span>
          <Link
            href="/admin"
            className="text-slate-600 hover:text-slate-900 font-medium"
          >
            Admin Telemetry
          </Link>
        </div>
      </div>
    </div>
  );
}
