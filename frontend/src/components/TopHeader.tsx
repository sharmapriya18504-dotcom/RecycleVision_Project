'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Menu,
  ScanLine,
  ChevronRight,
  ShieldCheck,
  Recycle,
  Sparkles,
  KeyRound,
  Check,
  X,
  MessageSquare
} from 'lucide-react';
import { getStoredOpenAIKey, setStoredOpenAIKey } from '@/lib/apiKeyStorage';

interface TopHeaderProps {
  onToggleSidebar: () => void;
  backendHealthy: boolean | null;
}

export const TopHeader: React.FC<TopHeaderProps> = ({
  onToggleSidebar,
  backendHealthy,
}) => {
  const pathname = usePathname();
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [currentKey, setCurrentKey] = useState('');
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    setCurrentKey(getStoredOpenAIKey());
    const handleKeyChange = () => setCurrentKey(getStoredOpenAIKey());
    window.addEventListener('recyclevision_key_changed', handleKeyChange);
    return () => window.removeEventListener('recyclevision_key_changed', handleKeyChange);
  }, []);

  const handleSaveKey = (e: React.FormEvent) => {
    e.preventDefault();
    setStoredOpenAIKey(currentKey);
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      setApiKeyModalOpen(false);
    }, 1200);
  };

  const handleClearKey = () => {
    setCurrentKey('');
    setStoredOpenAIKey('');
    setApiKeyModalOpen(false);
  };

  const getPageTitle = (path: string) => {
    switch (path) {
      case '/':
        return { title: 'Dashboard Overview', category: 'Main' };
      case '/classify':
        return { title: 'Classify Waste Image', category: 'Inference Studio' };
      case '/chat':
        return { title: 'RecycleBot AI Assistant (ChatGPT)', category: 'Conversational AI' };
      case '/analytics':
        return { title: 'Analytics & Trends', category: 'Intelligence' };
      case '/guide':
        return { title: 'Recycling Knowledge Guide', category: 'Resources' };
      case '/performance':
        return { title: 'Model Performance Center', category: 'Evaluation' };
      case '/history':
        return { title: 'Prediction History', category: 'Audit Logs' };
      case '/about':
        return { title: 'About Project & Viva Notes', category: 'Documentation' };
      case '/admin':
        return { title: 'Administrator Telemetry', category: 'Administration' };
      case '/login':
        return { title: 'Authentication Gateway', category: 'Security' };
      default:
        return { title: 'RecycleVision AI', category: 'App' };
    }
  };

  const current = getPageTitle(pathname);

  return (
    <>
      <header className="sticky top-0 z-30 h-16 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        {/* Left: Mobile hamburger & Breadcrumb */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-100 lg:hidden focus:outline-hidden"
            aria-label="Toggle Navigation Drawer"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div>
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-slate-400">
              <span>RecycleVision</span>
              <ChevronRight className="w-3 h-3 text-slate-300" />
              <span className="text-emerald-700 font-semibold">{current.category}</span>
            </div>
            <h2 className="text-sm sm:text-base font-bold text-slate-900 tracking-tight">
              {current.title}
            </h2>
          </div>
        </div>

        {/* Right: Quick shortcuts & Health */}
        <div className="flex items-center gap-2.5">
          {/* OpenAI Key Configuration Button */}
          <button
            onClick={() => setApiKeyModalOpen(true)}
            className={`hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all ${
              currentKey
                ? 'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
            title="Configure OpenAI API Key for ChatGPT Omni-Prediction & Chat"
          >
            <Sparkles className="w-3.5 h-3.5 text-purple-600" />
            <span>{currentKey ? 'ChatGPT Key Active' : 'ChatGPT Key'}</span>
          </button>

          {/* Connection status pill */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
              backendHealthy === true
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : backendHealthy === false
                ? 'bg-rose-50 text-rose-700 border-rose-200'
                : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendHealthy === true
                  ? 'bg-emerald-500 animate-pulse'
                  : backendHealthy === false
                  ? 'bg-rose-500'
                  : 'bg-amber-500'
              }`}
            />
            <span className="hidden sm:inline font-mono">
              {backendHealthy === true ? 'AI Engine Ready' : 'AI Offline'}
            </span>
          </div>

          {/* Quick Chat shortcut */}
          {pathname !== '/chat' && (
            <Link
              href="/chat"
              className="hidden md:inline-flex items-center gap-1.5 bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all active:scale-98"
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Ask AI</span>
            </Link>
          )}

          {/* Quick classify button */}
          {pathname !== '/classify' && (
            <Link
              href="/classify"
              className="inline-flex items-center gap-1.5 bg-emerald-700 hover:bg-emerald-800 text-white px-3.5 py-1.5 rounded-xl text-xs font-semibold shadow-xs hover:shadow-sm transition-all active:scale-98"
            >
              <ScanLine className="w-3.5 h-3.5" />
              <span>Classify</span>
            </Link>
          )}
        </div>
      </header>

      {/* OpenAI Key Configuration Modal */}
      {apiKeyModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200 relative animate-in fade-in zoom-in-95 duration-200">
            <button
              onClick={() => setApiKeyModalOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 p-1 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2.5 mb-3 text-purple-700">
              <div className="w-9 h-9 rounded-xl bg-purple-100 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 text-base">OpenAI ChatGPT Key</h3>
                <p className="text-xs text-slate-500">Universal Omni-Vision & RecycleBot Chat</p>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed mb-4">
              Enter your personal OpenAI API Key to classify <strong>ANY</strong> waste or object (e-waste, batteries, textiles, medical waste) and chat live with RecycleBot.
              Your key is saved locally in your browser.
            </p>

            <form onSubmit={handleSaveKey} className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  API Key (sk-...)
                </label>
                <div className="relative">
                  <KeyRound className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                  <input
                    type="password"
                    placeholder="sk-proj-..."
                    value={currentKey}
                    onChange={(e) => setCurrentKey(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-xs font-mono rounded-xl border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
              </div>

              {savedSuccess && (
                <div className="flex items-center gap-2 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-3 py-2 rounded-xl">
                  <Check className="w-4 h-4 text-emerald-600" />
                  <span>Key saved successfully!</span>
                </div>
              )}

              <div className="flex items-center justify-between pt-2">
                {currentKey ? (
                  <button
                    type="button"
                    onClick={handleClearKey}
                    className="text-xs text-rose-600 hover:text-rose-700 font-medium underline"
                  >
                    Clear Key
                  </button>
                ) : <span />}

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setApiKeyModalOpen(false)}
                    className="px-3.5 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 text-xs font-semibold text-white bg-purple-600 hover:bg-purple-700 rounded-xl shadow-xs transition-all"
                  >
                    Save Key
                  </button>
                </div>
              </div>
            </form>

            <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-400 leading-normal">
              💡 <em>Note:</em> You can also set <code>OPENAI_API_KEY</code> permanently in <code>backend/.env</code>.
            </div>
          </div>
        </div>
      )}
    </>
  );
};
