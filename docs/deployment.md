# RecycleVision AI — Production Deployment Guide

This comprehensive guide walks you through deploying **RecycleVision AI** to production using **Supabase** (Database & Auth), **Render** (Python FastAPI Backend), and **Vercel** (Next.js Frontend).

---

## Deployment Architecture Overview

```
 [Vercel] Next.js Frontend (https://recyclevision-ai.vercel.app)
    │
    ├── HTTPS REST API ──> [Render] FastAPI AI Service (https://recyclevision-api.onrender.com)
    │                                │
    └── Supabase JS SDK ──┐          └── PostgreSQL / Storage
                          ▼                      │
             [Supabase] Cloud Auth & DB <────────┘
```

---

## Step 1: Set Up Supabase Database & Authentication

1. Go to [https://supabase.com](https://supabase.com) and create a free account.
2. Click **New Project** and name it `recyclevision-ai`. Set a secure database password and choose your nearest region.
3. Once the database is provisioned, navigate to the **SQL Editor** in the left sidebar.
4. Click **New Query**, open `database/migrations/001_initial_schema.sql` from this repository, copy its entire contents, paste it into the SQL editor, and click **Run**.
5. This creates the required tables (`profiles`, `predictions`, `model_versions`, `activity_logs`), performance indexes, and Row Level Security (RLS) policies.
6. Navigate to **Project Settings** -> **API** and copy:
   - **Project URL** (e.g., `https://xyzcompany.supabase.co`)
   - **anon / public key** (Used by the frontend)
   - **service_role key** (Used *strictly* by backend server-side operations if needed; NEVER expose this in browser code).

---

## Step 2: Deploy the FastAPI Backend to Render

Render is an ideal hosting service for Python applications requiring C-extension dependencies like TensorFlow.

### Option A: Standard Render Web Service (Recommended)
1. Push your repository to GitHub.
2. Sign in to [https://render.com](https://render.com) and click **New +** -> **Web Service**.
3. Connect your GitHub repository `RecycleVision_Project`.
4. Fill in the deployment details:
   - **Name**: `recyclevision-api`
   - **Root Directory**: leave blank or set to `backend`
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users (e.g., Frankfurt or Singapore)
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
     ```
5. In the **Environment Variables** section, add:
   - `MODEL_PATH` = `backend/models/best_model.h5`
   - `MAX_UPLOAD_MB` = `10`
   - `ALLOWED_ORIGINS` = `http://localhost:3000,https://recyclevision-ai.vercel.app` *(update with your real Vercel URL once deployed)*
   - `SUPABASE_URL` = *(Your Supabase Project URL)*
   - `SUPABASE_KEY` = *(Your Supabase anon key or service role key)*
   - `ADMIN_SECRET_KEY` = *(A strong secret string for admin operations)*
6. Click **Create Web Service**.

> [!TIP]
> **Free Tier RAM Management**: Render's free tier provides 512MB RAM. Our `backend/requirements.txt` specifically uses `tensorflow-cpu==2.12.0` on Linux, which keeps RAM usage around ~220–280MB. Avoid configuring multi-worker Uvicorn (`--workers 4`) on the free tier to prevent out-of-memory errors.

### Option B: Deploy via Docker (Alternative)
Render also supports Docker builds directly from our provided `backend/Dockerfile`:
1. In Render, select **Docker** as the environment.
2. Render will automatically detect `backend/Dockerfile` and assemble the lightweight Debian container.

---

## Step 3: Verify the Backend Deployment

Once Render finishes building:
1. Open your assigned Render URL: `https://<your-render-subdomain>.onrender.com/health`
2. You should receive:
   ```json
   {
     "status": "healthy",
     "service": "RecycleVision AI",
     "version": "1.0.0",
     "model_loaded": true,
     "timestamp": "..."
   }
   ```
3. Open `https://<your-render-subdomain>.onrender.com/docs` to test interactive Swagger API endpoints.

---

## Step 4: Deploy the Next.js Frontend to Vercel

1. Sign in to [https://vercel.com](https://vercel.com) using your GitHub account.
2. Click **Add New...** -> **Project**.
3. Select your `RecycleVision_Project` repository.
4. Configure the project settings:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click *Edit* and select `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
5. In **Environment Variables**, add:
   - `NEXT_PUBLIC_API_BASE_URL` = `https://<your-render-subdomain>.onrender.com` *(Do NOT include trailing slash)*
   - `NEXT_PUBLIC_SUPABASE_URL` = *(Your Supabase Project URL)*
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = *(Your Supabase Anon Key)*
6. Click **Deploy**. Vercel will install dependencies and compile the production build in under 90 seconds.

---

## Step 5: Final Cross-Origin & Security Configuration

1. Copy your live Vercel domain (e.g., `https://recyclevision-ai.vercel.app`).
2. Return to your Render dashboard -> **Environment Variables**.
3. Update `ALLOWED_ORIGINS` to include your Vercel domain:
   ```
   ALLOWED_ORIGINS=http://localhost:3000,https://recyclevision-ai.vercel.app
   ```
4. Save changes. Render will automatically perform a fast zero-downtime redeploy.
5. In Supabase Dashboard -> **Authentication** -> **URL Configuration**, add your Vercel URL as a valid **Site URL** and **Redirect URL**.

---

## Step 6: End-to-End Verification Checklist

- [ ] Upload an image on `/classify` and confirm prediction result displays within 2 seconds.
- [ ] Verify confidence percentage and top-3 breakdown appear.
- [ ] Confirm the item is logged to `/history` and can be exported as a CSV.
- [ ] Check `/analytics` to see the category distribution chart and daily trend updates.
- [ ] Navigate to `/performance` to view verified evaluation metrics and confusion matrix.
- [ ] Log in to `/admin` using your `ADMIN_SECRET_KEY` and verify server health metrics.
