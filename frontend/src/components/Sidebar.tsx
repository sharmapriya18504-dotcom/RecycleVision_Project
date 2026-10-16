'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  ScanLine,
  BarChart3,
  BookOpen,
  Award,
  History,
  ShieldCheck,
  Info,
  Recycle,
  Sparkles,
  CheckCircle2,
  X,
  ChevronRight,
  ExternalLink,
  MessageSquare
} from 'lucide-react';
import { CREATOR_INFO } from '@/lib/constants';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  backendHealthy: boolean | null;
}

interface NavLinkItem {
  href: string;
  label: string;
  icon: any;
  badge?: string;
  badgeColor?: string;
}

interface NavGroup {
  group: string;
  links: NavLinkItem[];
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  backendHealthy,
}) => {
  const pathname = usePathname();

  const navGroups: NavGroup[] = [
    {
      group: 'MAIN MENU',
      links: [
        { href: '/', label: 'Dashboard Overview', icon: LayoutDashboard },
        {
          href: '/classify',
          label: 'Classify Waste',
          icon: ScanLine,
          badge: 'AI Core',
          badgeColor: 'bg-emerald-600 text-white',
        },
        {
          href: '/chat',
          label: 'RecycleBot AI Chat',
          icon: MessageSquare,
          badge: 'GPT-4o',
          badgeColor: 'bg-purple-600 text-white',
        },
      ],
    },
    {
      group: 'ANALYTICS & AI',
      links: [
        { href: '/analytics', label: 'Analytics & Trends', icon: BarChart3 },
        { href: '/performance', label: 'Model Performance', icon: Award, badge: '78.5%' },
        { href: '/history', label: 'Prediction History', icon: History },
      ],
    },
    {
      group: 'KNOWLEDGE & ADMIN',
      links: [
        { href: '/guide', label: 'Recycling Guide', icon: BookOpen },
        { href: '/about', label: 'Project & Viva Notes', icon: Info },
        { href: '/admin', label: 'Admin Telemetry', icon: ShieldCheck },
      ],
    },
  ];

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-slate-900/50 backdrop-blur-xs lg:hidden transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-white border-r border-slate-200 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full lg:shadow-none'
        }`}
      >
        <div className="flex flex-col flex-1 overflow-y-auto">
          {/* Brand Header */}
          <div className="h-16 px-5 border-b border-slate-100 flex items-center justify-between">
            <Link href="/" onClick={onClose} className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-600 to-green-700 flex items-center justify-center text-white shadow-md shadow-emerald-700/20 group-hover:scale-105 transition-transform">
                <Recycle className="w-5 h-5 text-emerald-100 animate-spin-slow" />
              </div>
              <div>
                <span className="text-base font-bold tracking-tight text-slate-900 flex items-center gap-1">
                  Recycle<span className="text-emerald-600">Vision</span>
                  <span className="text-[10px] bg-emerald-100 text-emerald-800 font-extrabold px-1.5 py-0.2 rounded-md">
                    AI
                  </span>
                </span>
                <span className="text-[10px] block font-medium text-slate-400 -mt-0.5">
                  Waste Classifier & Insights
                </span>
              </div>
            </Link>

            {/* Mobile Close Button */}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 lg:hidden"
              aria-label="Close Sidebar"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Quick Action Button */}
          <div className="p-3.5 pb-2">
            <Link
              href="/classify"
              onClick={onClose}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-600 to-green-700 hover:from-emerald-700 hover:to-green-800 text-white py-2.5 px-3 rounded-xl text-xs font-bold shadow-sm hover:shadow-md transition-all active:scale-98"
            >
              <ScanLine className="w-4 h-4" />
              <span>Scan & Classify Photo</span>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-5 flex-1">
            {navGroups.map((group) => (
              <div key={group.group} className="space-y-1">
                <div className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  {group.group}
                </div>
                {group.links.map((link) => {
                  const Icon = link.icon;
                  const isActive = pathname === link.href;

                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      onClick={onClose}
                      className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all group ${
                        isActive
                          ? 'bg-emerald-50 text-emerald-800 font-semibold shadow-2xs'
                          : 'text-slate-600 hover:text-emerald-700 hover:bg-slate-50'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <Icon
                          className={`w-4 h-4 transition-colors ${
                            isActive
                              ? 'text-emerald-600'
                              : 'text-slate-400 group-hover:text-emerald-600'
                          }`}
                        />
                        <span>{link.label}</span>
                      </div>

                      {link.badge && (
                        <span
                          className={`text-[9px] font-bold px-1.5 py-0.5 rounded-md ${
                            link.badgeColor || 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {link.badge}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </div>
            ))}
          </nav>
        </div>

        {/* Sidebar Footer Widgets */}
        <div className="p-3.5 border-t border-slate-100 space-y-3 bg-slate-50/50">
          {/* AI Engine Status Card */}
          <div className="bg-white p-2.5 rounded-xl border border-slate-200/80 shadow-2xs space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">
                AI Engine
              </span>
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendHealthy === true
                      ? 'bg-emerald-500 animate-pulse'
                      : backendHealthy === false
                      ? 'bg-rose-500'
                      : 'bg-amber-500'
                  }`}
                />
                <span className="text-[10px] font-semibold text-slate-700">
                  {backendHealthy === true
                    ? 'Online'
                    : backendHealthy === false
                    ? 'Offline'
                    : 'Checking...'}
                </span>
              </div>
            </div>
            <div className="text-[10px] text-slate-500 font-mono flex items-center justify-between">
              <span>MobileNetV2</span>
              <span className="text-emerald-700 font-bold">78.5% Acc</span>
            </div>
          </div>

          {/* Student/Creator Profile Snippet */}
          <div className="px-1 flex items-center justify-between text-[11px] text-slate-500">
            <div>
              <span className="font-semibold text-slate-800 block text-xs">
                {CREATOR_INFO.name}
              </span>
              <span className="text-[10px] text-slate-400">B.Tech CSE Project</span>
            </div>
            <Link
              href="/about"
              onClick={onClose}
              className="text-emerald-700 hover:text-emerald-800 font-semibold text-[10px] underline"
            >
              Info
            </Link>
          </div>
        </div>
      </aside>
    </>
  );
};
