'use client';

import React, { useState } from 'react';
import {
  BookOpen,
  Search,
  CheckCircle2,
  AlertTriangle,
  Info,
  ShieldCheck,
  Recycle,
  Sparkles
} from 'lucide-react';
import { STATIC_RECYCLING_GUIDES, CATEGORY_META, WASTE_CLASSES } from '@/lib/constants';
import { WasteClass } from '@/lib/types';
import { CategoryBadge } from '@/components/CategoryBadge';

export default function GuidePage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCategories = WASTE_CLASSES.filter((cat) => {
    if (selectedCategory !== 'all' && selectedCategory !== cat) return false;
    if (!searchQuery.trim()) return true;

    const query = searchQuery.toLowerCase();
    const guide = STATIC_RECYCLING_GUIDES[cat];
    const meta = CATEGORY_META[cat];

    const matchName = meta.label.toLowerCase().includes(query);
    const matchSummary = guide.summary.toLowerCase().includes(query);
    const matchExamples = guide.common_examples.some((ex) => ex.toLowerCase().includes(query));
    const matchTips = guide.recycling_tips.some((tip) => tip.toLowerCase().includes(query));
    const matchWarnings = guide.contamination_warnings.some((w) => w.toLowerCase().includes(query));

    return matchName || matchSummary || matchExamples || matchTips || matchWarnings;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 inline-flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5" />
          Material Segregation Standards
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Recycling Knowledge Guide
        </h1>
        <p className="text-sm text-slate-600">
          Learn how to properly prepare, clean, and segregate each waste material to eliminate batch contamination.
        </p>
      </div>

      {/* Search & Filter Bar */}
      <div className="bg-white p-4 sm:p-5 rounded-3xl border border-slate-200 shadow-xs flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search items (e.g., pizza box, foil)..."
            className="w-full pl-10 pr-4 py-2 text-xs rounded-xl border border-slate-200 focus:outline-hidden focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 bg-slate-50/50"
          />
        </div>

        {/* Category Pills */}
        <div className="flex items-center gap-1.5 flex-wrap w-full md:w-auto">
          <button
            onClick={() => setSelectedCategory('all')}
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
              onClick={() => setSelectedCategory(cat)}
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

      {/* Guide Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {filteredCategories.map((category) => {
          const guide = STATIC_RECYCLING_GUIDES[category];
          const meta = CATEGORY_META[category];
          const isRecyclable = guide.is_recyclable;

          return (
            <div
              key={category}
              className="bg-white rounded-3xl border border-slate-200 shadow-xs hover:shadow-md transition-shadow p-6 sm:p-7 flex flex-col justify-between space-y-6"
            >
              <div className="space-y-5">
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <CategoryBadge category={category} size="lg" showRecyclableBadge />
                </div>

                <p className="text-xs sm:text-sm text-slate-600 leading-relaxed font-medium">
                  {guide.summary}
                </p>

                {/* Common Examples */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                    Common Examples:
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {guide.common_examples.map((ex) => (
                      <span
                        key={ex}
                        className="text-xs bg-slate-50 text-slate-700 px-2.5 py-1 rounded-lg border border-slate-200"
                      >
                        {ex}
                      </span>
                    ))}
                  </div>
                </div>

                {/* How to dispose */}
                <div className="bg-emerald-50/40 rounded-2xl p-4 border border-emerald-100 space-y-1.5">
                  <h4 className="text-xs font-bold text-emerald-900 uppercase tracking-wide flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    Proper Disposal Protocol
                  </h4>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {guide.disposal_guide}
                  </p>
                </div>

                {/* Practical tips */}
                <div className="space-y-1.5">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                    <Sparkles className="w-3.5 h-3.5 text-amber-500" />
                    Practical Recycling Tips:
                  </span>
                  <ul className="space-y-1.5 text-xs text-slate-600">
                    {guide.recycling_tips.map((tip, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-emerald-600 font-bold">✓</span>
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Contamination warnings */}
                <div className="bg-rose-50/50 rounded-2xl p-4 border border-rose-100 space-y-1.5">
                  <h4 className="text-xs font-bold text-rose-900 uppercase tracking-wide flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4 text-rose-600" />
                    Contamination Pitfalls to Avoid
                  </h4>
                  <ul className="space-y-1 text-xs text-slate-700">
                    {guide.contamination_warnings.map((warn, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-rose-500 font-bold">•</span>
                        <span>{warn}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Local rules reminder */}
              <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-400 italic">
                {guide.local_rules_note}
              </div>
            </div>
          );
        })}
      </div>

      {filteredCategories.length === 0 && (
        <div className="bg-white rounded-3xl p-12 text-center border border-slate-200 text-slate-500 text-sm space-y-2">
          <p>No categories matching &quot;{searchQuery}&quot; were found.</p>
          <button
            onClick={() => {
              setSearchQuery('');
              setSelectedCategory('all');
            }}
            className="text-xs font-semibold text-emerald-700 underline"
          >
            Clear Search Filters
          </button>
        </div>
      )}
    </div>
  );
}
