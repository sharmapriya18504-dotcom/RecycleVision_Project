'use client';

import React from 'react';
import Link from 'next/link';
import {
  Info,
  BookOpen,
  Award,
  Layers,
  Cpu,
  Target,
  Sparkles,
  ShieldAlert,
  ArrowRight,
  Database,
  Globe,
  Terminal,
  CheckCircle2
} from 'lucide-react';
import { CREATOR_INFO } from '@/lib/constants';

export default function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Page Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <span className="text-xs font-bold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-3.5 py-1.5 rounded-full border border-emerald-200 inline-flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5" />
          Academic Project Documentation
        </span>
        <h1 className="text-3xl sm:text-5xl font-black text-slate-900 tracking-tight">
          About RecycleVision AI
        </h1>
        <p className="text-base text-slate-600 leading-relaxed">
          Comprehensive project report covering theoretical foundations, transfer-learning deep learning methodology, dataset curation, system architecture, and empirical evaluation.
        </p>
      </div>

      {/* Creator Profile Callout */}
      <div className="bg-gradient-to-r from-emerald-800 to-green-900 text-white rounded-3xl p-8 sm:p-10 shadow-xl flex flex-col sm:flex-row items-center gap-6">
        <div className="w-20 h-20 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 flex items-center justify-center font-black text-2xl text-emerald-300 shrink-0 shadow-inner">
          PS
        </div>
        <div className="space-y-1.5 text-center sm:text-left">
          <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
            Project Author & Lead Engineer
          </span>
          <h2 className="text-2xl font-bold">{CREATOR_INFO.name}</h2>
          <p className="text-sm text-emerald-100 font-medium">{CREATOR_INFO.degree}</p>
          <p className="text-xs text-emerald-200/90 leading-relaxed pt-1 max-w-xl">
            This application was conceived, engineered, and evaluated as a B.Tech Computer Science & Engineering portfolio project demonstrating deep learning model deployment, REST API integration, and user-centric environmental software.
          </p>
        </div>
      </div>

      {/* Structured Sections */}
      <div className="space-y-10 text-slate-700 text-sm sm:text-base leading-relaxed">
        {/* Section 1: Problem Statement & Objectives */}
        <section className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
            <Target className="w-4 h-4" />
            Chapter 1
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
            Problem Statement & Objectives
          </h3>
          <p>
            Municipal solid waste management is an urgent global environmental challenge. More than 2 billion tonnes of municipal solid waste are generated globally every year, with at least 33 percent of that not managed in an environmentally safe manner. In conventional waste handling, recyclable materials (such as aluminum cans, corrugated cardboard, and PET plastic bottles) are frequently commingled with non-recyclable refuse.
          </p>
          <p>
            Manual waste segregation at material recovery facilities (MRFs) is slow, dangerous for human workers, and prone to sorting errors. Contamination of single-stream recycling batches—such as greasy pizza boxes placed in cardboard recycling or mirrors thrown with glass jars—frequently ruins entire batches, diverting recyclable resources to landfills.
          </p>
          <div className="bg-emerald-50/60 rounded-2xl p-4 border border-emerald-100 space-y-2 text-xs text-emerald-950">
            <span className="font-bold block">Key Project Objectives:</span>
            <ul className="space-y-1 list-disc list-inside">
              <li>Train a high-performance deep neural network to classify common waste images into 6 categories.</li>
              <li>Employ Transfer Learning to achieve competitive accuracy on modest dataset sizes without severe overfitting.</li>
              <li>Engineer a decoupled, asynchronous FastAPI backend capable of real-time image preprocessing and inference.</li>
              <li>Create a responsive Next.js web application providing immediate predictions, probability distributions, and educational recycling guidance.</li>
              <li>Persist classification history with Supabase PostgreSQL and maintain live environmental analytics.</li>
            </ul>
          </div>
        </section>

        {/* Section 2: Dataset Overview */}
        <section className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
            <Database className="w-4 h-4" />
            Chapter 2
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
            Dataset Curation & Preprocessing
          </h3>
          <p>
            The model was trained on the benchmark <strong>TrashNet dataset</strong>, created by Mindy Yang and Gary Thung at Stanford University. The dataset consists of 2,527 high-resolution RGB photographs of single waste objects captured under consistent white backgrounds with varying natural illumination and angles.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-center text-xs">
            {[
              { cat: 'Cardboard', count: 403 },
              { cat: 'Glass', count: 501 },
              { cat: 'Metal', count: 410 },
              { cat: 'Paper', count: 594 },
              { cat: 'Plastic', count: 482 },
              { cat: 'Trash', count: 137 },
            ].map((c) => (
              <div key={c.cat} className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="font-semibold text-slate-800 block">{c.cat}</span>
                <span className="font-mono text-emerald-700 font-bold">{c.count} images</span>
              </div>
            ))}
          </div>
          <h4 className="font-bold text-sm text-slate-900 pt-2">Data Augmentation & Normalization</h4>
          <p className="text-xs sm:text-sm">
            To prevent overfitting on the 2,527 images, we applied real-time data augmentation using Keras <code className="bg-slate-100 px-1 rounded-sm text-xs font-mono">ImageDataGenerator</code>:
          </p>
          <ul className="text-xs space-y-1 list-disc list-inside text-slate-600">
            <li>Rotation range: ±20 degrees</li>
            <li>Zoom range: 20% random zoom</li>
            <li>Horizontal flip: Enabled</li>
            <li>Pixel rescaling: Multiplied by 1/255.0 to map intensities into floating-point range [0.0, 1.0]</li>
            <li>Image dimension resizing: Bilinear interpolation to 224 × 224 × 3</li>
            <li>Split: 80% Training subset (2,024 images) and 20% Held-Out Validation/Test subset (503 images)</li>
          </ul>
        </section>

        {/* Section 3: Model Architecture & Transfer Learning */}
        <section className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
            <Cpu className="w-4 h-4" />
            Chapter 3
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
            Model Architecture & Transfer Learning
          </h3>
          <p>
            Rather than training a standard Convolutional Neural Network (CNN) from scratch—which requires millions of labeled samples to learn low-level visual filters—we utilized <strong>Transfer Learning</strong> with <strong>MobileNetV2</strong>.
          </p>
          <p>
            MobileNetV2 is an efficient convolutional neural network pre-trained on ImageNet (1.4 million images across 1,000 object classes). Its architecture introduces two fundamental improvements:
          </p>
          <ol className="text-xs sm:text-sm space-y-2 list-decimal list-inside text-slate-700">
            <li>
              <strong>Depthwise Separable Convolutions:</strong> Splits standard convolutions into a 3×3 depthwise convolution (filtering one channel at a time) followed by a 1×1 pointwise convolution (linearly combining channel features). This reduces computational cost by approximately 85% compared to standard convolution layers.
            </li>
            <li>
              <strong>Inverted Residuals with Linear Bottlenecks:</strong> Shortcuts connect thin bottleneck layers rather than expanded representations, significantly minimizing peak memory consumption while preserving rich non-linear representations.
            </li>
          </ol>
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
            <span className="font-bold text-slate-900 block">Custom Head Architecture:</span>
            <div className="font-mono text-emerald-800 space-y-1">
              <div>Input Layer: (None, 224, 224, 3)</div>
              <div>↓ MobileNetV2 Backbone (155 layers, 2,257,984 frozen parameters)</div>
              <div>↓ GlobalAveragePooling2D (reduces 7×7×1280 feature maps into 1280-dim vector)</div>
              <div>↓ Dense(128, activation=&apos;relu&apos;, 163,968 trainable parameters)</div>
              <div>↓ Dense(6, activation=&apos;softmax&apos;, 774 trainable parameters)</div>
            </div>
            <p className="text-slate-500 pt-1">
              Total Parameters: 2,422,726 | Trainable Parameters: 164,742 (6.8%)
            </p>
          </div>
        </section>

        {/* Section 4: Evaluation Results */}
        <section className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
            <Award className="w-4 h-4" />
            Chapter 4
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
            Empirical Evaluation & Results
          </h3>
          <p>
            The model was evaluated using strict machine learning evaluation standards on the 20% held-out test split (503 unseen images). The empirical metrics produced are:
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs">
            <div className="p-3 bg-emerald-50 rounded-xl border border-emerald-200">
              <span className="text-slate-500 block">Test Accuracy</span>
              <span className="text-xl font-black font-mono text-emerald-700">78.53%</span>
            </div>
            <div className="p-3 bg-teal-50 rounded-xl border border-teal-200">
              <span className="text-slate-500 block">Macro F1-Score</span>
              <span className="text-xl font-black font-mono text-teal-700">0.7522</span>
            </div>
            <div className="p-3 bg-sky-50 rounded-xl border border-sky-200">
              <span className="text-slate-500 block">Weighted F1</span>
              <span className="text-xl font-black font-mono text-sky-700">0.7810</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
              <span className="text-slate-500 block">Test Samples</span>
              <span className="text-xl font-black font-mono text-slate-800">503 Images</span>
            </div>
          </div>
          <p className="text-xs sm:text-sm">
            Cardboard and paper achieved the highest F1-scores (0.86 and 0.86), while the trash class exhibited lower recall due to the limited number of samples (137 images) in the dataset. Full confusion matrix and learning curves are visible in the{' '}
            <Link href="/performance" className="text-emerald-700 underline font-semibold">
              Model Performance Center
            </Link>.
          </p>
        </section>

        {/* Section 5: Real-World Applications & Future Scope */}
        <section className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
            <Globe className="w-4 h-4" />
            Chapter 5
          </div>
          <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
            Real-World Applications, Limitations & Future Scope
          </h3>
          <div className="space-y-3">
            <h4 className="font-bold text-sm text-slate-900">Potential Real-World Deployments:</h4>
            <ul className="text-xs space-y-1 list-disc list-inside text-slate-600">
              <li><strong>Smart Reverse Vending Machines (RVMs):</strong> Optical automated kiosks that refund deposits for aluminum cans and PET bottles.</li>
              <li><strong>Municipal Material Recovery Facilities (MRFs):</strong> Integration with robotic delta sorting arms for primary sorting streams.</li>
              <li><strong>Educational Mobile Applications:</strong> Empowering citizens to verify curbside recycling acceptability instantly using smartphone cameras.</li>
            </ul>

            <h4 className="font-bold text-sm text-slate-900 pt-2">System Limitations:</h4>
            <ul className="text-xs space-y-1 list-disc list-inside text-slate-600">
              <li><strong>Single-label classification:</strong> The current system predicts one category per photograph and assumes the waste item is centered and isolated.</li>
              <li><strong>Surface reflection sensitivity:</strong> Clear glass bottles and transparent PET bottles can exhibit overlapping optical characteristics under direct sunlight glare.</li>
              <li><strong>Municipal variance:</strong> Recycling regulations differ significantly between municipal corporations.</li>
            </ul>

            <h4 className="font-bold text-sm text-slate-900 pt-2">Planned Future Upgrades:</h4>
            <ul className="text-xs space-y-1 list-disc list-inside text-slate-600">
              <li><strong>YOLOv8 Multi-Object Detection:</strong> Training an object detector on bounding-box annotated waste datasets (such as TACO) to locate and classify multiple items in cluttered scenes.</li>
              <li><strong>Mobile Application via React Native:</strong> Native camera integration for offline on-device inference using TensorFlow Lite (TFLite).</li>
              <li><strong>Barcode & QR Scanner Integration:</strong> Scanning product barcodes to look up specific resin codes (#1 to #7) from manufacturer packaging databases.</li>
            </ul>
          </div>
        </section>
      </div>

      {/* Navigation CTA */}
      <div className="pt-4 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-200">
        <Link
          href="/classify"
          className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 inline-flex items-center gap-1.5"
        >
          ← Return to Waste Classification
        </Link>
        <Link
          href="/performance"
          className="text-xs font-semibold text-slate-700 hover:text-slate-900 inline-flex items-center gap-1.5"
        >
          Inspect Verified Model Metrics →
        </Link>
      </div>
    </div>
  );
}
