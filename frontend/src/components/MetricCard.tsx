import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: {
    value: string;
    isPositive?: boolean;
  };
  accentColor?: 'emerald' | 'blue' | 'amber' | 'teal' | 'slate';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  accentColor = 'emerald',
}) => {
  const colorMap = {
    emerald: {
      bg: 'bg-emerald-50/70',
      text: 'text-emerald-700',
      border: 'border-emerald-100',
      iconBg: 'bg-emerald-600 text-white',
    },
    teal: {
      bg: 'bg-teal-50/70',
      text: 'text-teal-700',
      border: 'border-teal-100',
      iconBg: 'bg-teal-600 text-white',
    },
    blue: {
      bg: 'bg-sky-50/70',
      text: 'text-sky-700',
      border: 'border-sky-100',
      iconBg: 'bg-sky-600 text-white',
    },
    amber: {
      bg: 'bg-amber-50/70',
      text: 'text-amber-700',
      border: 'border-amber-100',
      iconBg: 'bg-amber-600 text-white',
    },
    slate: {
      bg: 'bg-slate-50',
      text: 'text-slate-700',
      border: 'border-slate-200',
      iconBg: 'bg-slate-700 text-white',
    },
  };

  const scheme = colorMap[accentColor] || colorMap.emerald;

  return (
    <div
      className={`bg-white rounded-2xl p-6 border ${scheme.border} shadow-xs hover:shadow-md transition-shadow relative overflow-hidden`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
            {title}
          </p>
          <h3 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            {value}
          </h3>
        </div>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-xs ${scheme.iconBg}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>

      {(subtitle || trend) && (
        <div className="mt-3 flex items-center justify-between text-xs pt-3 border-t border-slate-100">
          {subtitle && <span className="text-slate-500">{subtitle}</span>}
          {trend && (
            <span
              className={`font-semibold px-2 py-0.5 rounded-full ${
                trend.isPositive
                  ? 'bg-emerald-50 text-emerald-700'
                  : 'bg-rose-50 text-rose-700'
              }`}
            >
              {trend.value}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
