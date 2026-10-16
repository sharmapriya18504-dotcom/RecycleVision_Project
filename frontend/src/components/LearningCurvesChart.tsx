'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { TrainingEpochMetric } from '@/lib/types';

interface LearningCurvesChartProps {
  history: TrainingEpochMetric[];
}

export const LearningCurvesChart: React.FC<LearningCurvesChartProps> = ({ history }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
      {/* Accuracy Curves */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs">
        <h4 className="text-sm font-bold text-slate-900 mb-1">
          Training vs Validation Accuracy
        </h4>
        <p className="text-xs text-slate-500 mb-4">
          Convergence curve across 15 epochs of MobileNetV2 transfer learning
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis domain={[0.5, 1.0]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                  fontSize: '12px',
                }}
                formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`]}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Line
                type="monotone"
                dataKey="train_accuracy"
                name="Training Accuracy"
                stroke="#10b981"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                name="Validation Accuracy"
                stroke="#0284c7"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Loss Curves */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-xs">
        <h4 className="text-sm font-bold text-slate-900 mb-1">
          Categorical Cross-Entropy Loss
        </h4>
        <p className="text-xs text-slate-500 mb-4">
          Reduction in loss objective function for training and validation splits
        </p>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1.3]} tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  border: '1px solid #e2e8f0',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
                  fontSize: '12px',
                }}
                formatter={(val: any) => [Number(val).toFixed(3)]}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Line
                type="monotone"
                dataKey="train_loss"
                name="Training Loss"
                stroke="#f59e0b"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="Validation Loss"
                stroke="#e11d48"
                strokeWidth={2.5}
                dot={{ r: 3 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
