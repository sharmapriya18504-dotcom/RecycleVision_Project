'use client';

import React from 'react';
import Link from 'next/link';
import { Recycle, Heart, ExternalLink, ShieldCheck, BookOpen } from 'lucide-react';
import { CREATOR_INFO } from '@/lib/constants';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-900 text-slate-300 border-t border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Info */}
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
                <Recycle className="w-5 h-5 text-emerald-100" />
              </div>
              <span className="text-xl font-bold tracking-tight text-white">
                Recycle<span className="text-emerald-400">Vision AI</span>
              </span>
            </div>
            <p className="text-sm text-slate-400 max-w-md leading-relaxed">
              Smart Waste Classification & Recycling Analytics. An end-to-end full-stack computer vision application powered by transfer learning with MobileNetV2, FastAPI, Next.js, and Supabase PostgreSQL.
            </p>
            <div className="flex items-center gap-2 text-xs text-slate-400 pt-1">
              <span className="inline-block px-2 py-0.5 rounded-sm bg-slate-800 text-emerald-400 font-mono">
                TensorFlow 2.12
              </span>
              <span className="inline-block px-2 py-0.5 rounded-sm bg-slate-800 text-teal-400 font-mono">
                FastAPI
              </span>
              <span className="inline-block px-2 py-0.5 rounded-sm bg-slate-800 text-sky-400 font-mono">
                Next.js 15
              </span>
              <span className="inline-block px-2 py-0.5 rounded-sm bg-slate-800 text-green-400 font-mono">
                Supabase
              </span>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-3">
              Application Pages
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/" className="hover:text-emerald-400 transition-colors">
                  Home Overview
                </Link>
              </li>
              <li>
                <Link href="/classify" className="hover:text-emerald-400 transition-colors">
                  Classify Waste Image
                </Link>
              </li>
              <li>
                <Link href="/analytics" className="hover:text-emerald-400 transition-colors">
                  Analytics Dashboard
                </Link>
              </li>
              <li>
                <Link href="/guide" className="hover:text-emerald-400 transition-colors">
                  Recycling Knowledge Guide
                </Link>
              </li>
              <li>
                <Link href="/performance" className="hover:text-emerald-400 transition-colors">
                  Model Performance Center
                </Link>
              </li>
              <li>
                <Link href="/history" className="hover:text-emerald-400 transition-colors">
                  Prediction History
                </Link>
              </li>
            </ul>
          </div>

          {/* Project & Creator */}
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 mb-3">
              Project & Creator
            </h4>
            <div className="bg-slate-800/70 rounded-xl p-3.5 border border-slate-700/60 space-y-2">
              <div className="text-sm font-semibold text-white">{CREATOR_INFO.name}</div>
              <div className="text-xs text-emerald-400 font-medium">{CREATOR_INFO.degree}</div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Academic portfolio & viva demonstration project exploring automated waste segregation and computer vision.
              </p>
              <div className="pt-1 flex items-center gap-3 text-xs">
                <Link
                  href="/about"
                  className="text-emerald-400 hover:text-emerald-300 font-medium inline-flex items-center gap-1"
                >
                  <BookOpen className="w-3.5 h-3.5" />
                  Documentation
                </Link>
                <Link
                  href="/admin"
                  className="text-slate-400 hover:text-white font-medium inline-flex items-center gap-1"
                >
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Admin
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-10 pt-6 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-3">
          <div>
            &copy; 2026 RecycleVision AI. Built for environmental sustainability and technical learning.
          </div>
          <div className="flex items-center gap-1 text-slate-400">
            Designed with <Heart className="w-3.5 h-3.5 text-rose-500 fill-rose-500 inline mx-0.5" /> by Priya Sharma
          </div>
        </div>
      </div>
    </footer>
  );
};
