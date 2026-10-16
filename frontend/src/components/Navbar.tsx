'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Recycle,
  ScanLine,
  BarChart3,
  BookOpen,
  Award,
  History,
  ShieldCheck,
  Info,
  Menu,
  X,
  Activity,
  UserCheck
} from 'lucide-react';
import { api } from '@/lib/api';

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const checkStatus = async () => {
      try {
        const res = await api.checkHealth();
        if (mounted) setBackendHealthy(res.status === 'healthy' && res.model_loaded);
      } catch {
        if (mounted) setBackendHealthy(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navLinks = [
    { href: '/', label: 'Home', icon: Recycle },
    { href: '/classify', label: 'Classify Waste', icon: ScanLine },
    { href: '/analytics', label: 'Analytics', icon: BarChart3 },
    { href: '/guide', label: 'Recycling Guide', icon: BookOpen },
    { href: '/performance', label: 'Performance', icon: Award },
    { href: '/history', label: 'History', icon: History },
    { href: '/about', label: 'About', icon: Info },
    { href: '/admin', label: 'Admin', icon: ShieldCheck },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-emerald-100 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-600 to-green-700 flex items-center justify-center text-white shadow-md shadow-emerald-700/20 group-hover:scale-105 transition-transform">
              <Recycle className="w-5 h-5 text-emerald-100 animate-spin-slow" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight text-slate-900 group-hover:text-emerald-800 transition-colors">
                Recycle<span className="text-emerald-600">Vision</span>
              </span>
              <span className="text-[10px] block font-semibold uppercase tracking-wider text-emerald-700/70 -mt-1">
                AI Analytics
              </span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-800 font-semibold shadow-xs'
                      : 'text-slate-600 hover:text-emerald-700 hover:bg-slate-50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-600' : 'text-slate-400'}`} />
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Right Status & Actions */}
          <div className="hidden sm:flex items-center gap-3">
            {/* Health Status Indicator */}
            <div
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                backendHealthy === true
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : backendHealthy === false
                  ? 'bg-rose-50 text-rose-700 border-rose-200'
                  : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}
              title={
                backendHealthy === true
                  ? 'FastAPI AI Backend Connected & Ready'
                  : backendHealthy === false
                  ? 'Backend disconnected or loading'
                  : 'Checking AI status...'
              }
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
              <span className="hidden md:inline">
                {backendHealthy === true
                  ? 'AI Online'
                  : backendHealthy === false
                  ? 'AI Offline'
                  : 'Checking AI'}
              </span>
            </div>

            {/* Quick Action Button */}
            <Link
              href="/classify"
              className="inline-flex items-center gap-1.5 bg-emerald-700 hover:bg-emerald-800 text-white px-3.5 py-1.5 rounded-lg text-sm font-semibold shadow-sm hover:shadow-md transition-all active:scale-98"
            >
              <ScanLine className="w-4 h-4" />
              Classify
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <div className="lg:hidden flex items-center gap-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 focus:outline-hidden"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-b border-emerald-100 bg-white px-4 pt-2 pb-4 space-y-1 shadow-lg">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-base font-medium transition-colors ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-800 font-semibold'
                    : 'text-slate-600 hover:text-emerald-700 hover:bg-slate-50'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-emerald-600' : 'text-slate-400'}`} />
                {link.label}
              </Link>
            );
          })}
          <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendHealthy ? 'bg-emerald-500' : 'bg-rose-500'
                }`}
              />
              {backendHealthy ? 'AI Engine Ready' : 'AI Engine Offline'}
            </div>
            <Link
              href="/classify"
              onClick={() => setMobileMenuOpen(false)}
              className="bg-emerald-700 text-white px-4 py-1.5 rounded-lg text-sm font-medium"
            >
              Start Classifying
            </Link>
          </div>
        </div>
      )}
    </header>
  );
};
