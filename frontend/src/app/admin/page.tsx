'use client';

import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  Lock,
  Cpu,
  Database,
  Activity,
  Layers,
  Sparkles,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  LogOut,
  Terminal,
  Clock
} from 'lucide-react';
import { MetricCard } from '@/components/MetricCard';
import { api } from '@/lib/api';
import { AdminStats, ActivityLog } from '@/lib/types';
import { formatDate } from '@/lib/utils';

export default function AdminPage() {
  const [adminKey, setAdminKey] = useState<string>('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [logs, setLogs] = useState<ActivityLog[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seedMessage, setSeedMessage] = useState<string | null>(null);

  // Check stored key in session
  useEffect(() => {
    const saved = sessionStorage.getItem('recyclevision_admin_key');
    const defaultDevKey = process.env.NEXT_PUBLIC_ADMIN_KEY || 'recyclevision-admin-dev-secret-key-2026';
    const keyToTry = saved || defaultDevKey;

    if (keyToTry) {
      setAdminKey(keyToTry);
      verifyAndLoad(keyToTry);
    }
  }, []);

  const verifyAndLoad = async (key: string) => {
    try {
      setIsLoading(true);
      setError(null);
      const [statsRes, logsRes] = await Promise.all([
        api.getAdminStats(key),
        api.getAdminLogs(key, 25),
      ]);
      setStats(statsRes);
      setLogs(logsRes);
      setIsAuthenticated(true);
      sessionStorage.setItem('recyclevision_admin_key', key);
    } catch (err: any) {
      setError(err.message || 'Invalid administrator secret key.');
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!adminKey.trim()) return;
    verifyAndLoad(adminKey.trim());
  };

  const handleLogout = () => {
    sessionStorage.removeItem('recyclevision_admin_key');
    setIsAuthenticated(false);
    setStats(null);
    setLogs([]);
  };

  const handleSeedData = async () => {
    try {
      setIsLoading(true);
      setSeedMessage(null);
      const res = await api.seedDemoData(adminKey);
      setSeedMessage(res.message);
      // Reload stats
      verifyAndLoad(adminKey);
    } catch (err: any) {
      setError(err.message || 'Failed to seed demo data.');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="max-w-md mx-auto px-4 py-20 space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-slate-900 text-white flex items-center justify-center mx-auto shadow-md">
            <Lock className="w-6 h-6 text-emerald-400" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Administrator Gateway
          </h1>
          <p className="text-xs text-slate-500">
            Protected endpoint requiring server-side administrator authorization.
          </p>
        </div>

        <form
          onSubmit={handleLoginSubmit}
          className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-xl space-y-4"
        >
          {error && (
            <div className="bg-rose-50 border border-rose-200 text-rose-800 p-3 rounded-xl text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700 block">
              Admin Secret Key (X-Admin-Key)
            </label>
            <input
              type="password"
              value={adminKey}
              onChange={(e) => setAdminKey(e.target.value)}
              placeholder="Enter ADMIN_SECRET_KEY..."
              className="w-full text-xs px-3.5 py-2.5 rounded-xl border border-slate-200 focus:outline-hidden focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 font-mono bg-slate-50/50"
              required
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold py-3 rounded-xl transition-colors shadow-xs cursor-pointer flex items-center justify-center gap-2"
          >
            {isLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4 text-emerald-400" />}
            Authenticate Admin Session
          </button>

          <p className="text-[11px] text-slate-400 text-center">
            Default dev key: <code className="bg-slate-100 px-1.5 py-0.5 rounded-sm text-slate-700 font-mono">recyclevision-admin-dev-secret-key-2026</code>
          </p>
        </form>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200 mb-2">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            Authenticated Admin Environment
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            System Administration & Health
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Real-time server telemetry, database connection mode, and security audit logs.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={() => verifyAndLoad(adminKey)}
            disabled={isLoading}
            className="p-2.5 rounded-xl border border-slate-300 text-slate-600 hover:bg-slate-50 transition-colors"
            title="Refresh system status"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={handleLogout}
            className="inline-flex items-center gap-1.5 text-xs font-semibold bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 px-3.5 py-2.5 rounded-xl transition-colors cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign Out
          </button>
        </div>
      </div>

      {seedMessage && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-900 p-4 rounded-2xl text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{seedMessage}</span>
        </div>
      )}

      {/* System Resource Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          title="Server RSS Memory"
          value={stats ? `${stats.resources.rss_memory_mb} MB` : '...'}
          subtitle="Resident process RAM"
          icon={Cpu}
          accentColor="slate"
        />
        <MetricCard
          title="CPU Utilization"
          value={stats ? `${stats.resources.cpu_percent}%` : '...'}
          subtitle="Server core workload"
          icon={Activity}
          accentColor="emerald"
        />
        <MetricCard
          title="Database Mode"
          value={stats ? stats.database.mode : '...'}
          subtitle={`${stats?.database.total_records || 0} stored predictions`}
          icon={Database}
          accentColor="teal"
        />
        <MetricCard
          title="Model Status"
          value={stats?.model.loaded ? 'Online' : 'Unloaded'}
          subtitle={`MobileNetV2 (v${stats?.model.version || '1.0.0'})`}
          icon={Sparkles}
          accentColor="blue"
        />
      </div>

      {/* Administrative Operations Bar */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-7 shadow-xs space-y-4">
        <h3 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <Terminal className="w-5 h-5 text-emerald-600" />
          Administrative Actions
        </h3>
        <p className="text-xs text-slate-500">
          Quick utilities to populate demonstration records for portfolio evaluation or viva review.
        </p>

        <div className="pt-2 flex flex-wrap gap-3">
          <button
            onClick={handleSeedData}
            disabled={isLoading}
            className="inline-flex items-center gap-2 text-xs font-semibold bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-300 text-white px-5 py-2.5 rounded-xl shadow-xs transition-colors cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Seed Demo Classification Records
          </button>
          <a
            href={api.getExportCsvUrl()}
            download="all_predictions.csv"
            className="inline-flex items-center gap-2 text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 px-5 py-2.5 rounded-xl border border-slate-200 transition-colors"
          >
            Download Complete Audit CSV
          </a>
        </div>
      </div>

      {/* Audit Activity Logs Table */}
      <div className="bg-white rounded-3xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div className="space-y-1">
            <h3 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Clock className="w-5 h-5 text-emerald-600" />
              Security & Activity Audit Logs
            </h3>
            <p className="text-xs text-slate-500">
              Audit trail recorded by the server-side DatabaseService
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            {logs.length} events logged
          </span>
        </div>

        {logs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3 px-3">ID</th>
                  <th className="py-3 px-3">Action</th>
                  <th className="py-3 px-3">Details</th>
                  <th className="py-3 px-3">User</th>
                  <th className="py-3 px-3">IP Address</th>
                  <th className="py-3 px-3">Timestamp</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium">
                {logs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-3 px-3 font-mono text-slate-400">#{log.id}</td>
                    <td className="py-3 px-3 font-mono font-bold text-emerald-700">
                      {log.action}
                    </td>
                    <td className="py-3 px-3 text-slate-700 max-w-xs truncate">
                      {log.details || '—'}
                    </td>
                    <td className="py-3 px-3 text-slate-500">
                      {log.user_id || 'guest'}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-400">
                      {log.ip_address || '127.0.0.1'}
                    </td>
                    <td className="py-3 px-3 text-slate-500">
                      {formatDate(log.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-10 text-slate-400 text-xs">
            No audit events logged yet.
          </div>
        )}
      </div>
    </div>
  );
}
