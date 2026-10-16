'use client';

import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { UploadCloud, Image as ImageIcon, X, RefreshCw, AlertCircle, Sparkles } from 'lucide-react';

interface ImageDropzoneProps {
  onImageSelected: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  isLoading?: boolean;
}

export const ImageDropzone: React.FC<ImageDropzoneProps> = ({
  onImageSelected,
  selectedFile,
  onClear,
  isLoading = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync preview when selectedFile changes
  React.useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const validateAndProcessFile = (file: File) => {
    setErrorMsg(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];

    if (!validTypes.includes(file.type)) {
      setErrorMsg('Please upload a valid JPG, PNG, or WebP image.');
      return;
    }

    const maxBytes = 10 * 1024 * 1024; // 10MB
    if (file.size > maxBytes) {
      setErrorMsg('Image size exceeds 10MB. Please choose a smaller image.');
      return;
    }

    onImageSelected(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isLoading) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  // Helper to load sample test images
  const loadSampleImage = async (category: string, url: string) => {
    try {
      setErrorMsg(null);
      const res = await fetch(url);
      const blob = await res.blob();
      const sampleFile = new File([blob], `sample_${category}.jpg`, { type: 'image/jpeg' });
      onImageSelected(sampleFile);
    } catch {
      setErrorMsg('Failed to load sample image. Please upload a local image file.');
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Upload Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !previewUrl && !isLoading && fileInputRef.current?.click()}
        className={`relative rounded-3xl border-2 border-dashed transition-all cursor-pointer overflow-hidden ${
          isDragging
            ? 'border-emerald-500 bg-emerald-50/70 scale-[1.01]'
            : previewUrl
            ? 'border-emerald-200 bg-slate-50 cursor-default'
            : 'border-slate-300 hover:border-emerald-400 bg-white hover:bg-slate-50/70'
        } p-6 sm:p-10 text-center flex flex-col items-center justify-center min-h-[320px]`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/jpg"
          className="hidden"
          onChange={handleFileChange}
          disabled={isLoading}
        />

        {previewUrl ? (
          <div className="w-full flex flex-col items-center">
            <div className="relative max-h-[300px] w-auto max-w-full rounded-2xl overflow-hidden shadow-md border border-slate-200 bg-black/5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="Selected waste preview"
                className="max-h-[280px] w-auto object-contain mx-auto"
              />
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                }}
                disabled={isLoading}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/70 hover:bg-slate-900 text-white backdrop-blur-xs transition-colors shadow-sm"
                title="Remove image"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="mt-4 flex items-center gap-3">
              <span className="text-xs font-medium text-slate-600 bg-white px-3 py-1 rounded-full border border-slate-200 shadow-2xs">
                {selectedFile?.name} ({(selectedFile ? selectedFile.size / 1024 : 0).toFixed(1)} KB)
              </span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                disabled={isLoading}
                className="text-xs text-emerald-700 hover:text-emerald-800 font-semibold inline-flex items-center gap-1 hover:underline"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Replace Image
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center max-w-md">
            <div className="w-16 h-16 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-4 shadow-inner">
              <UploadCloud className="w-8 h-8" />
            </div>
            <h4 className="text-lg font-bold text-slate-800">
              Drag and drop your waste photo here
            </h4>
            <p className="text-sm text-slate-500 mt-1">
              or <span className="text-emerald-600 font-semibold underline">browse from your computer</span>
            </p>
            <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
              <span>Supports JPG, PNG, WebP</span>
              <span>•</span>
              <span>Max size 10MB</span>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {errorMsg && (
        <div className="flex items-center gap-2 text-xs text-rose-700 bg-rose-50 border border-rose-200 px-4 py-2.5 rounded-xl">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Quick Sample Selector */}
      {!previewUrl && (
        <div className="bg-slate-50/80 rounded-2xl p-4 border border-slate-200/70">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 mb-2.5">
            <Sparkles className="w-3.5 h-3.5 text-amber-500" />
            <span>Don&apos;t have an image ready? Test with curated samples:</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              {
                label: 'Cardboard Box',
                category: 'cardboard',
                url: 'https://images.unsplash.com/photo-1530587191325-3db32d826c18?auto=format&fit=crop&w=400&q=80',
              },
              {
                label: 'Glass Bottle',
                category: 'glass',
                url: 'https://images.unsplash.com/photo-1513542789411-b6a5d4f31634?auto=format&fit=crop&w=400&q=80',
              },
              {
                label: 'Soda Can (Metal)',
                category: 'metal',
                url: 'https://images.unsplash.com/photo-1581009146145-b5ef050c2e1e?auto=format&fit=crop&w=400&q=80',
              },
              {
                label: 'Plastic Bottle',
                category: 'plastic',
                url: 'https://images.unsplash.com/photo-1528323273322-d81458248d40?auto=format&fit=crop&w=400&q=80',
              },
            ].map((sample) => (
              <button
                key={sample.category}
                type="button"
                onClick={() => loadSampleImage(sample.category, sample.url)}
                disabled={isLoading}
                className="text-xs bg-white hover:bg-emerald-50 hover:text-emerald-800 hover:border-emerald-300 border border-slate-200 rounded-xl px-3 py-2 text-slate-700 font-medium transition-all shadow-2xs text-left flex items-center justify-between group active:scale-98"
              >
                <span>{sample.label}</span>
                <span className="text-[10px] text-slate-400 group-hover:text-emerald-600">→</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
