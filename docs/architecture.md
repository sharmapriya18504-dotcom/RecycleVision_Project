# RecycleVision AI — System Architecture & Technical Specifications

**Tagline**: Smart Waste Classification & Recycling Analytics  
**Lead Developer**: Priya Sharma (B.Tech Computer Science & Engineering)  
**System Version**: 1.0.0 (Production-Ready Release)  

---

## 1. High-Level System Architecture

RecycleVision AI follows a modern, decoupled **monorepo architecture** designed for cloud scalability, strict security separation, and maintainability.

```
                      +---------------------------------------+
                      |         CLIENT WEB APPLICATION        |
                      |   (Next.js 14/15 App Router + TS)     |
                      |    Tailwind CSS + Lucide + Recharts   |
                      |          Hosted on Vercel             |
                      +-------------------+-------------------+
                                          |
                        HTTPS (REST API)  |  Supabase Auth SDK (JWT)
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
+-------------------------------+               +-----------------------------------+
|      AI BACKEND SERVICE       |               |        SUPABASE PLATFORM          |
|  (FastAPI + Uvicorn + Python) |               |  (Managed PostgreSQL + Auth)      |
|    TensorFlow / MobileNetV2   |               |   - User Authentication (Auth)    |
|   Pillow Image Preprocessing  |               |   - Row-Level Security (RLS)      |
|   Hosted on Render / Container|               |   - Persistent Database Tables    |
+---------------+---------------+               +-----------------+-----------------+
                |                                                 |
                | SQLite Fallback Sync (when Supabase offline)     |
                +-------------------------------------------------+
```

---

## 2. Component Breakdown

### 2.1 Frontend Tier (`frontend/`)
- **Framework**: Next.js App Router with TypeScript.
- **Styling**: Tailwind CSS utilizing an eco-conscious color palette:
  - Primary: Forest Green (`#1b4332`, `#2d6a4f`)
  - Secondary: Mint Green (`#52b788`, `#74c69d`, `#d8f3dc`)
  - Background: Crisp White & Slate Off-White (`#f8fafc`)
  - Typography: Clean Inter & system font hierarchy
- **Data Visualization**: Recharts (Interactive category distribution, daily inference timeline).
- **Icons**: Lucide React for consistent icons.
- **State & Communication**:
  - Direct REST communication via typed API helper (`lib/api.ts`).
  - Automatic error boundary handling and toast feedback.
  - Image drag-and-drop processing with instant local preview before prediction.

### 2.2 AI Backend Tier (`backend/`)
- **Runtime**: Python 3.10.x with FastAPI and Uvicorn ASGI server.
- **Deep Learning Framework**: TensorFlow 2.12.0 with Keras.
- **Model Backbone**: MobileNetV2 pre-trained on ImageNet with frozen weights and a customized classification head:
  - Backbone: MobileNetV2 (`input_shape=(224, 224, 3)`, ImageNet weights, 155 frozen feature extraction layers)
  - Pooling: `GlobalAveragePooling2D()` (reduces 7x7x1280 feature maps into a 1280-dimensional feature vector)
  - Dense Hidden: `Dense(128, activation='relu')`
  - Output Head: `Dense(6, activation='softmax')`
- **Supported Waste Classes** (strictly ordered):
  1. `cardboard` (Index 0)
  2. `glass` (Index 1)
  3. `metal` (Index 2)
  4. `paper` (Index 3)
  5. `plastic` (Index 4)
  6. `trash` (Index 5)
- **Image Pipeline**:
  - Safe PIL decoding (`Image.open()`) with stream verification.
  - Dimension formatting to target `(224, 224, 3)` with bilinear interpolation.
  - Normalization: Exact scaling factor of `1.0 / 255.0` (mapping pixel intensities to `[0.0, 1.0]`).
  - Output confidence: Top-3 ranked candidate categories with calibrated percentages.
  - Knowledge Base: Authentic category-specific disposal tips, recycling feasibility, and contamination warnings.

### 2.3 Database & Authentication Tier (`database/`)
- **Primary Cloud DB**: Supabase PostgreSQL.
- **Local Fallback**: SQLite database (`backend/data/recyclevision.db`) for seamless offline evaluation and zero-configuration local runs.
- **Tables**:
  - `profiles`: Stores user IDs, email, full names, and authorization role (`admin` vs `user`).
  - `predictions`: Records inference events (unique ID, user ID, predicted class, confidence, top 3 ranking JSON, model version, ISO timestamp).
  - `model_versions`: Tracks registered machine learning model versions, accuracy, F1-scores, and status.
  - `activity_logs`: Audit trail for administrative actions, data exports, and demo seeding events.
- **Security & Authorization**:
  - Row Level Security (RLS) policies on all tables.
  - Server-side role verification for administrator endpoints (`X-Admin-Key` header or Supabase Admin JWT).
  - Zero leakage of service-role keys or database credentials to browser clients.

---

## 3. Data Flow Diagram

```
[User Browser]
      |
      | 1. Drag & Drop Image
      v
[ImageDropzone Component]
      |
      | 2. File validation (type: JPG/PNG, size <= 10MB)
      v
[HTTP POST /api/v1/predict]
      |
      | 3. FastAPI receives multipart/form-data
      v
[ClassifierService]
      |
      | 4. Decode bytes via PIL -> convert('RGB') -> resize((224, 224))
      | 5. Rescale array: arr / 255.0 -> expand_dims(axis=0)
      | 6. Keras model.predict(arr) -> Softmax probabilities (6 classes)
      | 7. Sort indices -> Top 3 categories + confidence scores
      | 8. Attach recycling knowledge guide
      v
[DatabaseService]
      |
      | 9. Save record to Supabase (or SQLite fallback)
      | 10. Write activity log
      v
[HTTP 200 JSON Response]
      |
      v
[PredictionResultCard]
      |
      | 11. Render prediction badge, progress bars, & recycling guidance
      v
[User Dashboard Updates]
```

---

## 4. Key Engineering Decisions

1. **Why MobileNetV2?**
   - MobileNetV2 uses depthwise separable convolutions and inverted residual bottlenecks, delivering high classification accuracy (78.5%+ on TrashNet) with only ~2.42M parameters.
   - Inference takes ~25–50 milliseconds on modern CPUs, making it ideal for cost-efficient hosting on Render without demanding expensive GPU infrastructure.

2. **Why Decouple Frontend & Backend?**
   - Vercel serverless functions have a 250MB uncompressed bundle limit and standard execution timeouts that make large C++ compiled machine learning libraries (TensorFlow) unstable.
   - Hosting the Python backend on a dedicated container service (Render, Railway, Fly.io) allows full control over Python dependencies, model caching in RAM, and continuous background processing.

3. **Dual-Mode Persistence (Supabase + Local SQLite)**:
   - For students, examiners, and evaluators running the project locally without an internet connection or Supabase API keys, the system falls back to SQLite.
   - When deployed to production, configuring `SUPABASE_URL` switches persistence to Supabase PostgreSQL without modifying a single line of application code.

4. **Dual-Engine Hybrid AI Architecture (MobileNetV2 + ChatGPT Vision)**:
   - **Engine 1: Local MobileNetV2 Core**: Instantaneous (~30ms), 100% offline, zero-cost inference for the 6 primary TrashNet benchmark categories (Cardboard, Glass, Metal, Paper, Plastic, Trash). Preserves the custom-trained model.
   - **Engine 2: ChatGPT Vision Omni-Mode (GPT-4o-mini)**: Universal multimodal perception capable of identifying *any* waste object under the sun—including complex electronic waste, swollen batteries, composite clothing, pharmaceutical blister packs, automotive parts, and industrial waste.
   - **RecycleBot AI Chat Assistant**: Conversational agent fluent in English and Hinglish to guide citizens through localized disposal rules, contamination prevention, and circular upcycling.
   - **Graceful Fallback**: If an OpenAI API key is not configured, the system provides an informative preview and continues to operate seamlessly on the local MobileNetV2 model without throwing unhandled exceptions.

