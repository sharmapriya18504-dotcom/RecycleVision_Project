-- ====================================================================
-- RecycleVision AI — Supabase PostgreSQL Database Schema Migration
-- Migration: 001_initial_schema.sql
-- ====================================================================

-- 1. PROFILES TABLE (Mirrors Supabase Auth Users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 2. MODEL VERSIONS TABLE
CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    architecture TEXT NOT NULL,
    accuracy DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    classes JSONB NOT NULL DEFAULT '["cardboard", "glass", "metal", "paper", "plastic", "trash"]'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 3. PREDICTIONS TABLE (Stores Inference Events)
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    predicted_class TEXT NOT NULL CHECK (predicted_class IN ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    top_predictions JSONB NOT NULL,
    image_url TEXT,
    model_version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 4. ACTIVITY LOGS (Admin and Operational Audit Trail)
CREATE TABLE IF NOT EXISTS public.activity_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    details TEXT,
    ip_address TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- ====================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ====================================================================
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON public.predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_class ON public.predictions(predicted_class);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON public.activity_logs(created_at DESC);

-- ====================================================================
-- AUTOMATIC PROFILE TRIGGER (ON NEW USER SIGN-UP)
-- ====================================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, role)
    VALUES (
        new.id,
        new.email,
        COALESCE(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name', split_part(new.email, '@', 1)),
        COALESCE(new.raw_user_meta_data->>'role', 'user')
    );
    RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();

-- ====================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ====================================================================
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activity_logs ENABLE ROW LEVEL SECURITY;

-- Profiles Policies
CREATE POLICY "Users can view their own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

CREATE POLICY "Admins have full access to profiles"
    ON public.profiles FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Model Versions Policies (Public Read, Admin Write)
CREATE POLICY "Anyone can view active model versions"
    ON public.model_versions FOR SELECT
    USING (true);

CREATE POLICY "Only admins can modify model versions"
    ON public.model_versions FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Predictions Policies
CREATE POLICY "Users can view own predictions"
    ON public.predictions FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Authenticated or anonymous users can insert predictions"
    ON public.predictions FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Admins have full access to predictions"
    ON public.predictions FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Activity Logs Policies (Admin Only View)
CREATE POLICY "Admins can view activity logs"
    ON public.activity_logs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

CREATE POLICY "Backend service can insert logs"
    ON public.activity_logs FOR INSERT
    WITH CHECK (true);

-- ====================================================================
-- INITIAL SEED DATA
-- ====================================================================
INSERT INTO public.model_versions (version, name, architecture, accuracy, f1_score, is_active)
VALUES (
    '1.0.0',
    'RecycleVision MobileNetV2 Core',
    'MobileNetV2 Transfer Learning with GlobalAveragePooling2D',
    0.9150,
    0.9120,
    true
)
ON CONFLICT (version) DO NOTHING;
