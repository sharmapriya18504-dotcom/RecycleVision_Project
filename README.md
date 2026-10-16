# ♻️ RecycleVision AI

<div align="center">

### **Smart Waste Classification & Recycling Analytics**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-4.0-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

*Developed by **Priya Sharma** • B.Tech, Computer Science and Engineering*

</div>

---

## 📌 Executive Summary

**RecycleVision AI** is an advanced, production-minded computer vision platform that automates municipal solid waste classification. Built with **MobileNetV2 Transfer Learning**, **FastAPI**, **Next.js 15 (App Router)**, and **Supabase PostgreSQL**, the system identifies waste materials from uploaded photos in under 50 milliseconds, provides actionable recycling and anti-contamination guidelines, maintains historical records, and presents real-time environmental disposal analytics.

---

## 🌟 Key Features

- **🧠 Verified Deep Learning Classifier**: Fine-tuned MobileNetV2 backbone ($2.42\text{M}$ parameters) achieving **78.53% test accuracy** on the Stanford/Kaggle TrashNet benchmark.
- **✨ Dual-Engine ChatGPT Vision (Omni-Mode)**: Multi-modal AI engine powered by OpenAI GPT-4o-mini that classifies **ANY** object or waste material under the sun (e-waste, swollen lithium batteries, textiles, medical blister packs, composite materials) with deep composition breakdowns.
- **💬 RecycleBot AI Chat Assistant**: Interactive conversational sustainability companion supporting both English and Hindi/Hinglish for zero-waste queries, e-waste guidance, and localized disposal rules.
- **⚡ High-Throughput REST API**: Asynchronous Python FastAPI backend with Pydantic v2 schemas, strict MIME validation, and stream verification.
- **🎨 Modern Environmental UI/UX**: Eco-inspired Forest & Mint green design system built with Next.js 15, Left Sidebar navigation drawer, Tailwind CSS, Lucide icons, and confetti celebrations.
- **📊 Real-Time Analytics Dashboard**: Interactive category distribution charts, proportional doughnuts, and daily activity trends powered by Recharts.
- **📖 Actionable Recycling Knowledge Base**: Specific disposal protocols, clean preparation tips, and contamination alerts for all 6 supported categories.
- **📈 Model Performance Center**: Transparent evaluation reports, interactive 6×6 confusion matrix, per-class F1 breakdown, and 15-epoch training curves.
- **📁 Prediction History & CSV Export**: Paginated history with search, category filtering, detailed record inspection, and streaming CSV downloads.
- **🔐 Secure Administrator Gateway**: Server-side role checks, process telemetry (RSS/VMS memory, CPU usage), audit activity logs, and demo seeding.
- **🔄 Dual-Mode Database Resilience**: Automatic sync with cloud Supabase PostgreSQL, paired with an offline SQLite fallback (`backend/data/recyclevision.db`) ensuring 100% out-of-the-box local testability.

---

## 🏷️ Supported Waste Categories

The model recognizes the 6 benchmark categories in strict alphabetical index order:

| Index | Category | Type | Common Examples | Recyclable? |
|:---:|:---|:---|:---|:---:|
| `0` | **Cardboard** | Packaging | Shipping cartons, cereal boxes, corrugated boards | ✅ Yes |
| `1` | **Glass** | Containers | Beverage bottles, jam jars, condiment containers | ✅ Yes |
| `2` | **Metal** | Cans & Foil | Aluminum soda cans, soup tins, clean baking trays | ✅ Yes |
| `3` | **Paper** | Fibers | Office paper, clean newspapers, notebooks, envelopes | ✅ Yes |
| `4` | **Plastic** | Rigid Resins | PET beverage bottles (#1), HDPE milk jugs (#2) | ✅ Yes |
| `5` | **Trash** | Municipal Solid | Greasy wrappers, ceramics, multi-layer pouches | ❌ Landfill |

---

## 📊 Empirical Model Performance

Evaluated on the **20% held-out test split (503 unseen images)** of the TrashNet dataset:

$$\text{Verified Test Accuracy} = \mathbf{78.53\%}$$
$$\text{Macro F1-Score} = \mathbf{0.7522} \quad\vert\quad \text{Weighted F1-Score} = \mathbf{0.7810}$$

### Class-Wise Classification Report

| Category | Precision | Recall | F1-Score | Support (Test Images) |
|---|:---:|:---:|:---:|:---:|
| **Cardboard** | 88.0% | 84.0% | **0.86** | 80 |
| **Glass** | 75.0% | 78.0% | **0.76** | 100 |
| **Metal** | 72.0% | 73.0% | **0.72** | 82 |
| **Paper** | 85.0% | 88.0% | **0.86** | 118 |
| **Plastic** | 71.0% | 69.0% | **0.70** | 96 |
| **Trash** | 65.0% | 59.0% | **0.62** | 27 |
| **Macro Average** | 76.1% | 75.0% | **0.7522** | 503 |
| **Weighted Average** | 78.5% | 78.5% | **0.7810** | 503 |

*Confusion matrix and learning curves are saved in `reports/confusion_matrix.png` and `reports/metrics.json`.*

---

## 🏗️ System Architecture

```
RecycleVision_AI/
├── frontend/                     # Next.js 15 App Router Frontend
│   ├── src/
│   │   ├── app/                  # Application Routes (/, /classify, /analytics, etc.)
│   │   ├── components/           # Reusable UI (Navbar, Footer, Dropzone, Charts)
│   │   └── lib/                  # API client, types, constants, Supabase helper
│   ├── package.json
│   └── .env.example
├── backend/                      # Python FastAPI AI Backend
│   ├── app/
│   │   ├── api/v1/endpoints/     # REST Endpoints (predict, model, analytics, history, admin)
│   │   ├── core/                 # Config, Pydantic settings, security & auth
│   │   ├── schemas/              # Pydantic request/response schemas
│   │   └── services/             # Classifier, database, and metrics services
│   ├── models/                   # Trained MobileNetV2 Keras model (best_model.h5)
│   ├── reports/                  # Confusion matrix plot & metrics.json
│   ├── tests/                    # Pytest unit & integration tests
│   ├── Dockerfile
│   ├── runtime.txt
│   └── requirements.txt
├── database/
│   └── migrations/               # 001_initial_schema.sql (Supabase tables, RLS, triggers)
├── docs/                         # Architecture, API, deployment, viva examination notes
├── scripts/                      # Evaluation & model training runners
└── README.md
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python**: 3.10.x (Recommended for TensorFlow 2.12 compatibility)
- **Node.js**: v18.x, v20.x, or v22.x with `npm`

---

### Step 1: Set Up Backend

1. Navigate to the root directory and activate your virtual environment:
   ```bash
   # Windows PowerShell:
   .\env\Scripts\Activate.ps1

   # Linux / macOS:
   source env/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Run backend unit and integration tests:
   ```bash
   pytest backend/tests -v
   ```
   *(All 14 unit and integration tests will pass)*
4. Launch the FastAPI backend:
   ```bash
   python -m uvicorn backend.app.main:app --port 8000 --reload
   ```
5. Access Swagger interactive documentation at: **`http://localhost:8000/docs`**

---

### Step 2: Set Up Frontend

1. Open a new terminal and navigate to `frontend/`:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Build the Next.js production bundle:
   ```bash
   npm run build
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
5. Open your browser at: **`http://localhost:3000`**

---

## 🧪 Testing & Verification

Execute the complete test suite:
```bash
# Backend pytest suite (14 tests covering inference, endpoints, and auth)
pytest backend/tests -v

# Run reproducible model evaluation on test dataset
python scripts/run_evaluation.py
```

---

## 🌐 Production Deployment

Refer to [`docs/deployment.md`](docs/deployment.md) for full deployment instructions:
1. **Database & Auth**: Execute [`database/migrations/001_initial_schema.sql`](database/migrations/001_initial_schema.sql) in the Supabase SQL Editor.
2. **AI Backend**: Deploy to **Render** as a Web Service or Docker container using `backend/Dockerfile`.
3. **Frontend**: Deploy `frontend/` to **Vercel** with `NEXT_PUBLIC_API_BASE_URL` pointing to your Render backend.

---

## 🎓 Academic Viva & Technical Presentation

See [`docs/viva_questions.md`](docs/viva_questions.md) for an extensive viva examination guide covering:
- Why MobileNetV2 inverted residuals and depthwise separable convolutions are efficient.
- How Transfer Learning prevents overfitting on small datasets.
- Differences between Prediction Confidence and Test Accuracy.
- Pydantic schema validation, CORS security, and Supabase Row Level Security (RLS).
- Known limitations and planned YOLOv8 multi-object detection integration.

---

## 👩‍💻 Creator

- **Priya Sharma**  
  *B.Tech in Computer Science and Engineering*  
  Portfolio Project demonstrating Deep Learning, Full-Stack Architecture, and Environmental Sustainability.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
