'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Recycle,
  Lock,
  Mail,
  ShieldCheck,
  ArrowRight,
  AlertCircle,
  CheckCircle2,
  Sparkles
} from 'lucide-react';
import { supabase, isSupabaseConfigured } from '@/lib/supabase';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [authSuccess, setAuthSuccess] = useState<string | null>(null);

  const handleSupabaseLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isSupabaseConfigured || !supabase) {
      setAuthError('Supabase is not configured yet with valid cloud credentials. Use Admin Key Login below.');
      return;
    }

    try {
      setIsSubmitting(true);
      setAuthError(null);
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) throw error;
      setAuthSuccess('Authenticated successfully! Redirecting...');
      setTimeout(() => router.push('/'), 1000);
    } catch (err: any) {
      setAuthError(err.message || 'Login failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAdminKeyLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (!adminKey.trim()) return;
    sessionStorage.setItem('recyclevision_admin_key', adminKey.trim());
    router.push('/admin');
  };

  return (
    <div className="max-w-md mx-auto px-4 py-16 space-y-8">
      {/* Brand Header */}
      <div className="text-center space-y-2">
        <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-600 to-green-700 text-white flex items-center justify-center mx-auto shadow-md">
          <Recycle className="w-6 h-6 text-emerald-100" />
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Welcome to RecycleVision
        </h1>
        <p className="text-xs text-slate-500">
          Sign in to access personal prediction history and administration tools.
        </p>
      </div>

      {authError && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-3.5 rounded-2xl text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{authError}</span>
        </div>
      )}

      {authSuccess && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-900 p-3.5 rounded-2xl text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{authSuccess}</span>
        </div>
      )}

      {/* Supabase Auth Form */}
      <form
        onSubmit={handleSupabaseLogin}
        className="bg-white p-6 sm:p-7 rounded-3xl border border-slate-200 shadow-sm space-y-4"
      >
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
            User Account Login
          </span>
          <span
            className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
              isSupabaseConfigured
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-amber-50 text-amber-700'
            }`}
          >
            {isSupabaseConfigured ? 'Supabase Auth' : 'Demo Mode'}
          </span>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-700 block">Email Address</label>
          <div className="relative">
            <Mail className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="user@example.com"
              className="w-full pl-9 pr-3 py-2 text-xs rounded-xl border border-slate-200 focus:outline-hidden focus:border-emerald-500 bg-slate-50/50"
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-700 block">Password</label>
          <div className="relative">
            <Lock className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full pl-9 pr-3 py-2 text-xs rounded-xl border border-slate-200 focus:outline-hidden focus:border-emerald-500 bg-slate-50/50"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-300 text-white text-xs font-bold py-2.5 rounded-xl transition-colors shadow-2xs cursor-pointer"
        >
          {isSubmitting ? 'Verifying...' : 'Sign In with Supabase'}
        </button>

        {!isSupabaseConfigured && (
          <p className="text-[11px] text-amber-700 bg-amber-50/70 p-2 rounded-lg border border-amber-200 leading-normal">
            ℹ️ Supabase environment variables not yet provided. Guests can classify waste and view analytics immediately without logging in.
          </p>
        )}
      </form>

      {/* Quick Admin Passkey Entry Card */}
      <form
        onSubmit={handleAdminKeyLogin}
        className="bg-slate-900 text-white p-6 rounded-3xl border border-slate-800 shadow-md space-y-4"
      >
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Quick Admin Passkey Entry
          </h3>
        </div>
        <p className="text-[11px] text-slate-400">
          Enter your <code className="text-emerald-300">ADMIN_SECRET_KEY</code> to immediately access telemetry, server logs, and demo seeding.
        </p>

        <input
          type="password"
          value={adminKey}
          onChange={(e) => setAdminKey(e.target.value)}
          placeholder="recyclevision-admin-dev-secret-key-2026"
          className="w-full px-3 py-2 text-xs rounded-xl border border-slate-700 bg-slate-950 text-white font-mono focus:outline-hidden focus:border-emerald-500"
        />

        <button
          type="submit"
          className="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold py-2 rounded-xl transition-colors"
        >
          Access Admin Dashboard →
        </button>
      </form>

      <div className="text-center">
        <Link
          href="/"
          className="text-xs text-slate-500 hover:text-slate-800 inline-flex items-center gap-1 font-medium"
        >
          ← Return to Home Overview
        </Link>
      </div>
    </div>
  );
}
