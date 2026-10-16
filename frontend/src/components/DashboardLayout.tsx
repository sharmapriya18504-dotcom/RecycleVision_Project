'use client';

import React, { useState, useEffect } from 'react';
import { Sidebar } from './Sidebar';
import { TopHeader } from './TopHeader';
import { api } from '@/lib/api';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
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
    const interval = setInterval(checkStatus, 25000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen flex bg-slate-50/60 text-slate-900 antialiased">
      {/* Fixed Left Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        backendHealthy={backendHealthy}
      />

      {/* Main Content Area (Offset by sidebar width on desktop) */}
      <div className="flex-1 flex flex-col min-w-0 lg:pl-64">
        <TopHeader
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          backendHealthy={backendHealthy}
        />

        <main className="flex-1 w-full">
          {children}
        </main>
      </div>
    </div>
  );
};
