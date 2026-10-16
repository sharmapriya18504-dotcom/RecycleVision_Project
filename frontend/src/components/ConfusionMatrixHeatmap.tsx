'use client';

import React from 'react';
import { WasteClass } from '@/lib/types';
import { CATEGORY_META } from '@/lib/constants';

interface ConfusionMatrixHeatmapProps {
  classes: WasteClass[];
  matrix: number[][];
}

export const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixHeatmapProps> = ({
  classes,
  matrix,
}) => {
  // Find maximum value to normalize opacity/color intensity
  let maxVal = 1;
  matrix.forEach((row) => {
    row.forEach((val) => {
      if (val > maxVal) maxVal = val;
    });
  });

  return (
    <div className="w-full overflow-x-auto pb-2">
      <div className="min-w-[500px]">
        {/* Header row: Predicted labels */}
        <div className="grid grid-cols-7 gap-1.5 mb-1.5 text-center">
          <div className="text-[11px] font-bold text-slate-400 self-end pb-1">
            Actual ↓ / Pred →
          </div>
          {classes.map((c) => (
            <div
              key={`col-${c}`}
              className="text-xs font-semibold capitalize text-slate-700 bg-slate-50 py-1.5 px-1 rounded-lg border border-slate-200"
            >
              {c}
            </div>
          ))}
        </div>

        {/* Matrix rows */}
        {classes.map((actualClass, rowIdx) => {
          const rowValues = matrix[rowIdx] || [];
          const rowTotal = rowValues.reduce((a, b) => a + b, 0);

          return (
            <div key={`row-${actualClass}`} className="grid grid-cols-7 gap-1.5 mb-1.5">
              {/* Actual label */}
              <div className="text-xs font-semibold capitalize text-slate-700 bg-slate-50 py-2 px-2 rounded-lg border border-slate-200 flex items-center justify-between">
                <span>{actualClass}</span>
                <span className="text-[10px] text-slate-400 font-mono">({rowTotal})</span>
              </div>

              {/* Cells */}
              {classes.map((predClass, colIdx) => {
                const count = rowValues[colIdx] || 0;
                const isDiagonal = rowIdx === colIdx;
                const intensity = maxVal > 0 ? count / maxVal : 0;

                // Color based on whether it is diagonal (correct True Positives)
                const bgColor = isDiagonal
                  ? `rgba(16, 185, 129, ${Math.max(intensity, 0.15)})`
                  : count > 0
                  ? `rgba(244, 63, 94, ${Math.max(intensity * 0.75, 0.08)})`
                  : 'rgba(248, 250, 252, 1)';

                const textColor =
                  intensity > 0.5 ? 'text-slate-900 font-bold' : 'text-slate-600 font-medium';

                return (
                  <div
                    key={`cell-${rowIdx}-${colIdx}`}
                    style={{ backgroundColor: bgColor }}
                    className={`py-3 px-2 rounded-xl text-center text-xs transition-transform hover:scale-105 border ${
                      isDiagonal ? 'border-emerald-300' : 'border-slate-100'
                    } ${textColor} flex flex-col items-center justify-center shadow-2xs`}
                    title={`Actual: ${actualClass}, Predicted: ${predClass} (${count} samples)`}
                  >
                    <span className="font-mono text-sm">{count}</span>
                    {rowTotal > 0 && (
                      <span className="text-[9px] text-slate-500 font-mono">
                        {((count / rowTotal) * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}

        <div className="mt-3 flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-sm bg-emerald-400 border border-emerald-500" />
            <span>Diagonal cells: Correct True Positives</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-sm bg-rose-200 border border-rose-300" />
            <span>Off-diagonal: Misclassified samples</span>
          </div>
        </div>
      </div>
    </div>
  );
};
