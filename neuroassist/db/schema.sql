-- ============================================================
-- schema.sql - NeuroAssist PostgreSQL DDL
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension unavailable (%). Using array fallback for embeddings.', SQLERRM;
  END;
END $$;

-- 1) Patients
CREATE TABLE IF NOT EXISTS patients (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  mrn            VARCHAR(50) UNIQUE,
  first_name     VARCHAR(100),
  last_name      VARCHAR(100),
  date_of_birth  DATE,
  sex            VARCHAR(10),
  created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- 2) Sessions
CREATE TABLE IF NOT EXISTS sessions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id  UUID REFERENCES patients(id) ON DELETE CASCADE,
  status      VARCHAR(20) DEFAULT 'active',
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 3) Patient context (JSONB diagnostic fields)
CREATE TABLE IF NOT EXISTS patient_context (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id          UUID REFERENCES patients(id) ON DELETE CASCADE,
  session_id          UUID REFERENCES sessions(id) ON DELETE CASCADE,
  symptoms            JSONB DEFAULT '{}',
  clinical            JSONB DEFAULT '{}',
  genomics            JSONB DEFAULT '{}',
  vasari              JSONB DEFAULT '{}',
  pathology           JSONB DEFAULT '{}',
  labs                JSONB DEFAULT '{}',
  treatment_history   JSONB DEFAULT '{}',
  fields_populated    JSONB DEFAULT '{}',
  completeness_score  DECIMAL(4,3),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 4) Messages
CREATE TABLE IF NOT EXISTS messages (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id    UUID REFERENCES sessions(id) ON DELETE CASCADE,
  role          VARCHAR(10) NOT NULL,
  content       TEXT NOT NULL,
  content_json  JSONB,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 5) Questions asked
CREATE TABLE IF NOT EXISTS questions_asked (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id       UUID REFERENCES sessions(id) ON DELETE CASCADE,
  question_id      VARCHAR(100),
  question_type    VARCHAR(5),
  category         VARCHAR(50),
  fields_targeted  JSONB,
  answer_ids       JSONB,
  answer_free_text TEXT,
  answered_at      TIMESTAMPTZ
);

-- 6) Imaging reports
CREATE TABLE IF NOT EXISTS imaging_reports (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id          UUID REFERENCES patients(id) ON DELETE CASCADE,
  session_id          UUID REFERENCES sessions(id) ON DELETE CASCADE,
  scan_filename       VARCHAR(500),
  scan_format         VARCHAR(10),
  scan_date           DATE,
  wt_volume_cm3       DECIMAL(8,3),
  ncr_volume_cm3      DECIMAL(8,3),
  ed_volume_cm3       DECIMAL(8,3),
  et_volume_cm3       DECIMAL(8,3),
  diameter_mm         DECIMAL(6,2),
  centroid_x          INTEGER,
  centroid_y          INTEGER,
  centroid_z          INTEGER,
  lobe_frontal        BOOLEAN,
  lobe_temporal       BOOLEAN,
  lobe_parietal       BOOLEAN,
  lobe_occipital      BOOLEAN,
  lobe_other          BOOLEAN,
  lobe_pct_frontal    DECIMAL(5,1),
  lobe_pct_temporal   DECIMAL(5,1),
  lobe_pct_parietal   DECIMAL(5,1),
  lobe_pct_occipital  DECIMAL(5,1),
  lobe_pct_other      DECIMAL(5,1),
  necrosis_ratio      DECIMAL(5,4),
  enhancement_ratio   DECIMAL(5,4),
  edema_ratio         DECIMAL(5,4),
  midline_shift_mm    DECIMAL(5,2),
  surface_to_volume   DECIMAL(6,4),
  annotated_dir       VARCHAR(500),
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 7) Diagnostic reports
CREATE TABLE IF NOT EXISTS diagnostic_reports (
  id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id              UUID REFERENCES patients(id) ON DELETE CASCADE,
  session_id              UUID REFERENCES sessions(id) ON DELETE CASCADE,
  imaging_report_id       UUID REFERENCES imaging_reports(id),
  who_grade_predicted     VARCHAR(10),
  diagnosis_label         TEXT,
  confidence_score        DECIMAL(4,3),
  data_completeness       DECIMAL(4,3),
  survival_category       VARCHAR(20),
  survival_score          INTEGER,
  estimated_median_months VARCHAR(20),
  factors_favorable       JSONB,
  factors_unfavorable     JSONB,
  treatment_flags         JSONB,
  full_report             JSONB,
  reviewed_by             VARCHAR(100),
  reviewed_at             TIMESTAMPTZ,
  created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- 8) Knowledge chunks (pgvector if available, otherwise array fallback)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    EXECUTE '
      CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source      VARCHAR(200),
        chunk_text  TEXT,
        embedding   vector(384),
        metadata    JSONB,
        created_at  TIMESTAMPTZ DEFAULT NOW()
      )';
  ELSE
    EXECUTE '
      CREATE TABLE IF NOT EXISTS knowledge_chunks (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source      VARCHAR(200),
        chunk_text  TEXT,
        embedding   DOUBLE PRECISION[],
        metadata    JSONB,
        created_at  TIMESTAMPTZ DEFAULT NOW()
      )';
  END IF;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ctx_patient ON patient_context(patient_id);
CREATE INDEX IF NOT EXISTS idx_ctx_session ON patient_context(session_id);
CREATE INDEX IF NOT EXISTS idx_q_session ON questions_asked(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_q_session_question ON questions_asked(session_id, question_id);
CREATE INDEX IF NOT EXISTS idx_img_patient ON imaging_reports(patient_id, scan_date);
CREATE INDEX IF NOT EXISTS idx_rep_patient ON diagnostic_reports(patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_rep_session ON diagnostic_reports(session_id);

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    EXECUTE '
      CREATE INDEX IF NOT EXISTS idx_vec_embed
      ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)';
  END IF;
END $$;
