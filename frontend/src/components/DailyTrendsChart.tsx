'use client';

import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { DailyTrend } from '@/lib/types';

interface DailyTrendsChartProps {
  data: DailyTrend[];
}

export const DailyTrendsChart: React.FC<DailyTrendsChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 text-center text-slate-500 text-sm">
        No daily trend activity recorded yet.
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-3xl border border-emerald-100/80 shadow-xs w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
        <div>
          <h4 className="text-sm font-bold text-slate-900">
            Prediction Activity Over Time
          </h4>
          <p className="text-xs text-slate-500">
            Daily classification frequency recorded by database
          </p>
        </div>
        <span className="text-xs font-mono bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-1 rounded-full self-start sm:self-auto">
          Past 30 Days
        </span>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
            <defs>
              <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11 }}
              tickFormatter={(val) => {
                try {
                  const parts = val.split('-');
                  return `${parts[1]}/${parts[2]}`;
                } catch {
                  return val;
                }
              }}
            />
            <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                border: '1px solid #e2e8f0',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                fontSize: '12px',
              }}
              formatter={(val: any) => [`${val} items`, 'Classified']}
            />
            <Area
              type="monotone"
              dataKey="count"
              stroke="#059669"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#trendGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
