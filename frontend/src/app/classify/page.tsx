'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  ScanLine,
  Sparkles,
  RefreshCw,
  AlertCircle,
  History,
  CheckCircle2,
  HelpCircle,
  ArrowRight,
  Zap,
  Globe2,
  KeyRound,
  MessageSquare
} from 'lucide-react';
import { ImageDropzone } from '@/components/ImageDropzone';
import { PredictionResultCard } from '@/components/PredictionResultCard';
import { OmniResultCard } from '@/components/OmniResultCard';
import { api } from '@/lib/api';
import { PredictionResponse, OmniPredictionResponse } from '@/lib/types';
import { getStoredOpenAIKey } from '@/lib/apiKeyStorage';

type ClassifierEngine = 'mobilenet' | 'chatgpt';

export default function ClassifyPage() {
  const [engine, setEngine] = useState<ClassifierEngine>('mobilenet');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [mobilenetResult, setMobilenetResult] = useState<PredictionResponse | null>(null);
  const [omniResult, setOmniResult] = useState<OmniPredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [userApiKey, setUserApiKey] = useState<string>('');

  useEffect(() => {
    setUserApiKey(getStoredOpenAIKey());
    const handleKeyChange = () => setUserApiKey(getStoredOpenAIKey());
    window.addEventListener('recyclevision_key_changed', handleKeyChange);
    return () => window.removeEventListener('recyclevision_key_changed', handleKeyChange);
  }, []);

  const handleImageSelected = (file: File) => {
    setSelectedFile(file);
    setMobilenetResult(null);
    setOmniResult(null);
    setErrorMessage(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setMobilenetResult(null);
    setOmniResult(null);
    setErrorMessage(null);
  };

  const handleClassify = async () => {
    if (!selectedFile) {
      setErrorMessage('Please choose or drag-and-drop a waste image first.');
      return;
    }

    try {
      setIsLoading(true);
      setErrorMessage(null);

      if (engine === 'mobilenet') {
        setLoadingStep('Uploading & validating image payload...');
        setTimeout(() => {
          setLoadingStep('Normalizing pixels & running MobileNetV2 inference...');
        }, 400);

        const result = await api.predictWaste(selectedFile);
        setMobilenetResult(result);
        setOmniResult(null);
      } else {
        setLoadingStep('Preparing image & sending base64 to ChatGPT Vision (GPT-4o)...');
        setTimeout(() => {
          setLoadingStep('Extracting materials, hazards, and specialized recycling protocols...');
        }, 800);

        const result = await api.predictOmniWaste(selectedFile, userApiKey || undefined);
        setOmniResult(result);
        setMobilenetResult(null);
      }
    } catch (err: any) {
      setErrorMessage(
        err.message || 'Failed to classify the image. Please verify that the AI backend is online.'
      );
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 inline-flex items-center gap-1.5">
          <ScanLine className="w-3.5 h-3.5" />
          Dual-Engine Inference Studio
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Classify Any Waste Material
        </h1>
        <p className="text-sm text-slate-600">
          Choose between our fast local deep learning model (6 TrashNet benchmark categories) or 
          <strong> ChatGPT Vision Omni-Mode</strong> to classify <em>literally any object</em> (e-waste, batteries, textiles, medical waste, etc.).
        </p>
      </div>

      {/* Dual Engine Selector Switch */}
      <div className="bg-slate-100 p-1.5 rounded-2xl flex flex-col sm:flex-row gap-2 max-w-2xl mx-auto border border-slate-200">
        <button
          onClick={() => {
            setEngine('mobilenet');
            setErrorMessage(null);
          }}
          className={`flex-1 py-3 px-4 rounded-xl font-bold text-xs sm:text-sm flex items-center justify-center gap-2.5 transition-all ${
            engine === 'mobilenet'
              ? 'bg-white text-emerald-800 shadow-sm border border-emerald-100'
              : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
          }`}
        >
          <Zap className={`w-4 h-4 ${engine === 'mobilenet' ? 'text-emerald-600' : 'text-slate-400'}`} />
          <span>MobileNetV2 (Core 6 Classes)</span>
          <span className="text-[10px] bg-emerald-100 text-emerald-800 font-mono px-2 py-0.5 rounded-full">
            Local 30ms
          </span>
        </button>

        <button
          onClick={() => {
            setEngine('chatgpt');
            setErrorMessage(null);
          }}
          className={`flex-1 py-3 px-4 rounded-xl font-bold text-xs sm:text-sm flex items-center justify-center gap-2.5 transition-all ${
            engine === 'chatgpt'
              ? 'bg-white text-purple-800 shadow-sm border border-purple-100'
              : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
          }`}
        >
          <Sparkles className={`w-4 h-4 ${engine === 'chatgpt' ? 'text-purple-600' : 'text-slate-400'}`} />
          <span>ChatGPT Vision (Omni-Mode)</span>
          <span className="text-[10px] bg-purple-100 text-purple-800 font-bold px-2 py-0.5 rounded-full">
            Any Item ✨
          </span>
        </button>
      </div>

      {/* Active Engine Notice */}
      <div
        className={`p-4 rounded-2xl text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border ${
          engine === 'mobilenet'
            ? 'bg-emerald-50/70 text-emerald-900 border-emerald-200'
            : 'bg-purple-50/70 text-purple-950 border-purple-200'
        }`}
      >
        <div className="flex items-center gap-2.5">
          {engine === 'mobilenet' ? (
            <Zap className="w-4 h-4 text-emerald-600 shrink-0" />
          ) : (
            <Sparkles className="w-4 h-4 text-purple-600 shrink-0" />
          )}
          <span>
            {engine === 'mobilenet'
              ? 'Local Model: Fast offline classification for Cardboard, Glass, Metal, Paper, Plastic, and Trash.'
              : 'ChatGPT Omni-Mode: Multi-modal GPT-4o vision identifies complex items (e.g. Li-ion batteries, cables, textiles, blister packs).'}
          </span>
        </div>

        {engine === 'chatgpt' && (
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-[11px] font-mono font-medium">
              {userApiKey ? '🔑 User Key Active' : '⚙️ Server / Demo Mode'}
            </span>
          </div>
        )}
      </div>

      {/* Main Upload Box & Action */}
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200/90 shadow-sm space-y-6">
        <ImageDropzone
          onImageSelected={handleImageSelected}
          selectedFile={selectedFile}
          onClear={handleClear}
          isLoading={isLoading}
        />

        {/* Action Button & Loading Status */}
        {selectedFile && !mobilenetResult && !omniResult && (
          <div className="flex flex-col items-center justify-center pt-2">
            <button
              onClick={handleClassify}
              disabled={isLoading}
              className={`w-full sm:w-auto inline-flex items-center justify-center gap-2.5 text-white px-8 py-4 rounded-2xl font-bold shadow-md hover:shadow-lg transition-all active:scale-98 cursor-pointer disabled:cursor-not-allowed ${
                engine === 'mobilenet'
                  ? 'bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-300'
                  : 'bg-purple-700 hover:bg-purple-800 disabled:bg-slate-300'
              }`}
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Processing Image...</span>
                </>
              ) : engine === 'mobilenet' ? (
                <>
                  <Zap className="w-5 h-5 text-emerald-200" />
                  <span>Predict with MobileNetV2</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 text-purple-200" />
                  <span>Classify ANY Item with ChatGPT Vision</span>
                </>
              )}
            </button>

            {isLoading && (
              <p
                className={`text-xs font-medium mt-3 animate-pulse ${
                  engine === 'mobilenet' ? 'text-emerald-700' : 'text-purple-700'
                }`}
              >
                {loadingStep}
              </p>
            )}
          </div>
        )}

        {/* Error Feedback */}
        {errorMessage && (
          <div className="bg-rose-50 border border-rose-200 rounded-2xl p-4 flex items-start gap-3 text-xs text-rose-800">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">Inference Request Failed</span>
              <p>{errorMessage}</p>
            </div>
          </div>
        )}
      </div>

      {/* Prediction Result Display */}
      {mobilenetResult && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              MobileNetV2 Classification Result
            </h2>
            <button
              onClick={handleClear}
              className="text-xs text-emerald-700 hover:text-emerald-800 font-semibold inline-flex items-center gap-1 hover:underline"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Classify Another Image
            </button>
          </div>

          <PredictionResultCard result={mobilenetResult} />
        </div>
      )}

      {omniResult && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">
              ChatGPT Vision Universal Prediction
            </h2>
            <button
              onClick={handleClear}
              className="text-xs text-purple-700 hover:text-purple-800 font-semibold inline-flex items-center gap-1 hover:underline"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Classify Another Image
            </button>
          </div>

          <OmniResultCard result={omniResult} />
        </div>
      )}

      {/* Quick Navigation Footer Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4">
        <Link
          href="/chat"
          className="bg-white hover:bg-purple-50/50 p-5 rounded-2xl border border-slate-200 hover:border-purple-200 shadow-2xs transition-all flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-100 text-purple-700 group-hover:bg-purple-600 group-hover:text-white transition-colors">
              <MessageSquare className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">Ask RecycleBot AI</h4>
              <p className="text-xs text-slate-500">Chat with ChatGPT about any item</p>
            </div>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-purple-600 group-hover:translate-x-1 transition-transform" />
        </Link>

        <Link
          href="/history"
          className="bg-white hover:bg-emerald-50/50 p-5 rounded-2xl border border-slate-200 hover:border-emerald-200 shadow-2xs transition-all flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-slate-100 text-slate-700 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
              <History className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">Prediction History</h4>
              <p className="text-xs text-slate-500">Inspect past classification events</p>
            </div>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-600 group-hover:translate-x-1 transition-transform" />
        </Link>

        <Link
          href="/guide"
          className="bg-white hover:bg-emerald-50/50 p-5 rounded-2xl border border-slate-200 hover:border-emerald-200 shadow-2xs transition-all flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-slate-100 text-slate-700 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
              <HelpCircle className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">Recycling Guide</h4>
              <p className="text-xs text-slate-500">Disposal criteria for all materials</p>
            </div>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-emerald-600 group-hover:translate-x-1 transition-transform" />
        </Link>
      </div>
    </div>
  );
}
