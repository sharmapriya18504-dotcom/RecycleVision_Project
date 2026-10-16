'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import {
  Sparkles,
  Send,
  Trash2,
  Bot,
  User,
  ShieldAlert,
  HelpCircle,
  ScanLine,
  RefreshCw,
  ExternalLink,
  KeyRound
} from 'lucide-react';
import { api } from '@/lib/api';
import { ChatMessage } from '@/lib/types';
import { getStoredOpenAIKey } from '@/lib/apiKeyStorage';

const SUGGESTED_PROMPTS = [
  'How do I safely dispose of a swollen lithium-ion battery?',
  'Can greasy pizza boxes or soiled paper be recycled?',
  'Purane phone chargers aur circuit boards kahan recycle karein?',
  'What is the difference between PET #1 and HDPE #2 plastics?',
  'How should expired prescription medicines and blister packs be disposed of?',
  'Purane kapde aur bedsheets ko upcycle kaise karein?'
];

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        '👋 **Namaste & Welcome to RecycleBot!** I am your sustainability & waste management AI companion powered by **ChatGPT**.\n\nAsk me anything about proper disposal, e-waste, hazardous materials, plastics, or zero-waste practices in English or Hindi!'
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userApiKey, setUserApiKey] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setUserApiKey(getStoredOpenAIKey());
    const handleKeyChange = () => setUserApiKey(getStoredOpenAIKey());
    window.addEventListener('recyclevision_key_changed', handleKeyChange);
    return () => window.removeEventListener('recyclevision_key_changed', handleKeyChange);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = textToSend || inputText;
    if (!text.trim() || isLoading) return;

    const userMsg: ChatMessage = { role: 'user', content: text.trim() };
    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInputText('');
    setIsLoading(true);

    try {
      // Send conversation to backend ChatGPT service
      const res = await api.chatWithRecycleBot(newHistory, userApiKey || undefined);
      setMessages([...newHistory, { role: 'assistant', content: res.reply }]);
    } catch (err: any) {
      setMessages([
        ...newHistory,
        {
          role: 'assistant',
          content: `⚠️ **Connection Error**: ${err.message || 'Could not communicate with the ChatGPT backend service. Please verify your connection or API key.'}`
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content:
          '🧹 Chat history cleared. How can I help you with recycling or waste management today?'
      }
    ]);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 space-y-6 flex flex-col h-[calc(100vh-6rem)]">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-white p-5 rounded-2xl border border-slate-200/90 shadow-2xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center text-white shadow-xs">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-900 tracking-tight">
                RecycleBot AI Assistant
              </h1>
              <span className="text-[10px] bg-purple-100 text-purple-800 font-bold px-2 py-0.5 rounded-full border border-purple-200">
                GPT-4o
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Interactive recycling, hazardous waste, and circular economy guide
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleClearChat}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-xl transition-colors text-xs font-semibold inline-flex items-center gap-1.5"
            title="Clear Chat History"
          >
            <Trash2 className="w-4 h-4" />
            <span className="hidden sm:inline">Clear Chat</span>
          </button>

          <Link
            href="/classify"
            className="px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200 rounded-xl text-xs font-semibold inline-flex items-center gap-1.5 transition-all"
          >
            <ScanLine className="w-3.5 h-3.5 text-emerald-600" />
            <span>Classify Waste</span>
          </Link>
        </div>
      </div>

      {/* Chat Messages Container */}
      <div className="flex-1 bg-white rounded-3xl border border-slate-200/90 p-4 sm:p-6 overflow-y-auto space-y-4 shadow-xs">
        {messages.map((msg, index) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={index}
              className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {!isUser && (
                <div className="w-8 h-8 rounded-xl bg-purple-100 text-purple-700 flex items-center justify-center shrink-0 mt-0.5">
                  <Sparkles className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 text-xs sm:text-sm leading-relaxed ${
                  isUser
                    ? 'bg-purple-700 text-white rounded-tr-xs shadow-xs'
                    : 'bg-slate-50 text-slate-800 border border-slate-200/80 rounded-tl-xs whitespace-pre-wrap'
                }`}
              >
                {msg.content}
              </div>

              {isUser && (
                <div className="w-8 h-8 rounded-xl bg-slate-900 text-white flex items-center justify-center shrink-0 mt-0.5">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}

        {isLoading && (
          <div className="flex gap-3 justify-start items-center">
            <div className="w-8 h-8 rounded-xl bg-purple-100 text-purple-700 flex items-center justify-center shrink-0">
              <Sparkles className="w-4 h-4 animate-spin" />
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-2xl px-4 py-2.5 text-xs text-slate-500 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-600 animate-ping" />
              <span>RecycleBot is typing recommendations...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Quick Prompts */}
      <div className="space-y-2">
        <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-400">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Quick Inquiries:</span>
        </div>
        <div className="flex gap-2 overflow-x-auto pb-1 no-scrollbar">
          {SUGGESTED_PROMPTS.map((prompt, i) => (
            <button
              key={i}
              onClick={() => handleSend(prompt)}
              disabled={isLoading}
              className="whitespace-nowrap px-3 py-1.5 rounded-full text-xs bg-white hover:bg-purple-50 text-slate-700 hover:text-purple-700 border border-slate-200 hover:border-purple-200 transition-all shrink-0 active:scale-95 disabled:opacity-50"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white p-2 rounded-2xl border border-slate-200/90 shadow-sm flex items-center gap-2">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask RecycleBot anything (e.g. 'Can I recycle blister packs?' or 'tube light kahan feke?')..."
          disabled={isLoading}
          className="flex-1 px-4 py-2.5 text-xs sm:text-sm bg-transparent focus:outline-hidden text-slate-900 placeholder:text-slate-400"
        />

        <button
          onClick={() => handleSend()}
          disabled={!inputText.trim() || isLoading}
          className="p-3 bg-purple-700 hover:bg-purple-800 disabled:bg-slate-200 text-white disabled:text-slate-400 rounded-xl transition-all cursor-pointer disabled:cursor-not-allowed shadow-xs active:scale-95"
          title="Send Message"
        >
          {isLoading ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
}
