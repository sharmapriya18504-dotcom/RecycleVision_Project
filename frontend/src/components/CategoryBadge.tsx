'use client';

import React from 'react';
import { WasteClass } from '@/lib/types';
import { CATEGORY_META } from '@/lib/constants';
import {
  Package,
  Wine,
  Layers,
  FileText,
  Milk,
  Trash2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

interface CategoryBadgeProps {
  category: WasteClass | string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  showRecyclableBadge?: boolean;
}

export const CategoryBadge: React.FC<CategoryBadgeProps> = ({
  category,
  size = 'md',
  showIcon = true,
  showRecyclableBadge = false,
}) => {
  const cat = (category?.toLowerCase() || 'trash') as WasteClass;
  const meta = CATEGORY_META[cat] || CATEGORY_META['trash'];

  const getIcon = () => {
    switch (cat) {
      case 'cardboard':
        return <Package className="w-3.5 h-3.5" />;
      case 'glass':
        return <Wine className="w-3.5 h-3.5" />;
      case 'metal':
        return <Layers className="w-3.5 h-3.5" />;
      case 'paper':
        return <FileText className="w-3.5 h-3.5" />;
      case 'plastic':
        return <Milk className="w-3.5 h-3.5" />;
      case 'trash':
      default:
        return <Trash2 className="w-3.5 h-3.5" />;
    }
  };

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-sm px-2.5 py-1 gap-1.5',
    lg: 'text-base px-3.5 py-1.5 gap-2 font-semibold',
  };

  const isRecyclable = cat !== 'trash';

  return (
    <div className="inline-flex items-center gap-1.5">
      <span
        className={`inline-flex items-center font-medium rounded-full border shadow-xs transition-colors ${meta.badgeClass} ${sizeClasses[size]}`}
      >
        {showIcon && getIcon()}
        <span className="capitalize">{meta.label}</span>
      </span>

      {showRecyclableBadge && (
        <span
          className={`inline-flex items-center text-xs px-2 py-0.5 rounded-full font-medium ${
            isRecyclable
              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
              : 'bg-rose-50 text-rose-700 border border-rose-200'
          }`}
        >
          {isRecyclable ? (
            <>
              <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-600" />
              Recyclable
            </>
          ) : (
            <>
              <AlertCircle className="w-3 h-3 mr-1 text-rose-600" />
              Landfill
            </>
          )}
        </span>
      )}
    </div>
  );
};
