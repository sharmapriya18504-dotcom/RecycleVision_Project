import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { DashboardLayout } from '@/components/DashboardLayout';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-sans',
});

export const metadata: Metadata = {
  title: 'RecycleVision AI — Smart Waste Classification & Recycling Analytics',
  description:
    'An advanced environmental AI application powered by MobileNetV2 transfer learning, FastAPI, Next.js, and Supabase. Classify garbage images, inspect prediction confidence, and explore recycling analytics.',
  keywords: [
    'waste classification',
    'deep learning',
    'recycling analytics',
    'MobileNetV2',
    'computer vision',
    'FastAPI',
    'Next.js',
    'TrashNet'
  ],
  authors: [{ name: 'Priya Sharma' }],
  creator: 'Priya Sharma',
  openGraph: {
    title: 'RecycleVision AI — Smart Waste Classification',
    description: 'AI-powered garbage image classification and recycling analytics.',
    siteName: 'RecycleVision AI',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased selection:bg-emerald-500 selection:text-white">
        <DashboardLayout>
          {children}
        </DashboardLayout>
      </body>
    </html>
  );
}
