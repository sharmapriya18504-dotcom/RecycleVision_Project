'use client';

import React from 'react';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { CategoryCount } from '@/lib/types';
import { CATEGORY_META } from '@/lib/constants';

interface CategoryDistributionChartProps {
  data: CategoryCount[];
}

export const CategoryDistributionChart: React.FC<CategoryDistributionChartProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-8 text-center text-slate-500 text-sm">
        No category distribution data available yet.
      </div>
    );
  }

  const chartData = data.map((d) => ({
    name: CATEGORY_META[d.category]?.label || d.category,
    category: d.category,
    count: d.count,
    percentage: d.percentage,
    fill: CATEGORY_META[d.category]?.color || '#10b981',
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
      {/* Bar Chart */}
      <div className="bg-white p-6 rounded-3xl border border-emerald-100/80 shadow-xs">
        <h4 className="text-sm font-bold text-slate-900 mb-1">
          Predictions by Waste Category
        </h4>
        <p className="text-xs text-slate-500 mb-5">
          Breakdown of total classified items recorded in database
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                  fontSize: '12px',
                }}
                formatter={(val: any) => [`${val} predictions`, 'Count']}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Doughnut Chart */}
      <div className="bg-white p-6 rounded-3xl border border-emerald-100/80 shadow-xs">
        <h4 className="text-sm font-bold text-slate-900 mb-1">
          Proportional Share
        </h4>
        <p className="text-xs text-slate-500 mb-2">
          Percentage distribution of all processed images
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                dataKey="count"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
                paddingAngle={3}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`pie-cell-${index}`} fill={entry.fill} stroke="#ffffff" strokeWidth={2} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e2e8f0',
                  fontSize: '12px',
                }}
                formatter={(val: any, name: any, item: any) => [
                  `${val} (${item.payload.percentage}%)`,
                  name,
                ]}
              />
              <Legend
                wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
                layout="horizontal"
                verticalAlign="bottom"
                align="center"
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
