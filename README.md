# Gioblastoma / NeuroAssist

NeuroAssist is a Flask-based clinical decision support prototype for brain tumor workflows. It combines MRI ingestion, 3D Attention U-Net segmentation, derived imaging metrics, structured neuro-oncology intake, lightweight RAG retrieval, and Groq-backed report generation into a single local project.

## What This Repo Contains

- `neuroassist/`: the main web application, routes, templates, config, database helpers, RAG utilities, and tests
- `data_pipeline.py`: MRI preprocessing, inference orchestration, tumor volume metrics, and lobe involvement logic
- `Load_model.py`: cached model loader for the trained segmentation checkpoint
- `get_labels.py`: exports annotated PNG slices and a summary montage from predicted masks
- `Input_preprocessing/`: standalone NIfTI<->DICOM conversion utilities
- `Tumor Model/`: trained model weights used by the segmentation pipeline
- `Data/`: local data workspace for imaging assets and experiments
- `utils/` and `gioblastoma.ipynb`: notebook-era experimentation helpers and research code

## End-To-End Flow

1. A clinician creates a session from the landing page.
2. The app accepts either NIfTI uploads or DICOM folder uploads.
3. DICOM uploads are converted to NIfTI when needed.
4. The segmentation model runs inference and computes:
   whole tumor, NCR, ED, ET volumes, diameter, centroid, and lobe involvement.
5. Annotated tumor slice PNGs are written under `neuroassist/static/annotated/`.
6. Imaging results are stored in PostgreSQL.
7. The clinician completes the structured intake form in the session view.
8. Completeness scoring, RAG lookup, and LLM report generation produce a structured final report.
9. The report view renders the persisted diagnostic summary plus annotated slices.

## Main Application Pieces

### Backend

- `neuroassist/app.py`: Flask app factory, runtime folder creation, DB bootstrap, blueprint registration
- `neuroassist/routes/session.py`: session creation, record directory, live session state for the UI
- `neuroassist/routes/upload.py`: MRI upload handling, DICOM conversion, inference, ingest into DB
- `neuroassist/routes/chat.py`: intake validation, completeness updates, RAG lookup, final report generation
- `neuroassist/routes/report.py`: report page, print view, annotated-slice gallery

### Services

- `neuroassist/services/llm.py`: Groq chat client, retry handling, JSON extraction, response normalization
- `neuroassist/services/context_builder.py`: combines patient profile, imaging metrics, context fields, and RAG chunks into prompt context
- `neuroassist/services/completeness.py`: weighted field coverage scoring for the intake form
- `neuroassist/services/rag.py`: embeddings + vector or lexical retrieval from `knowledge_chunks`
- `neuroassist/services/embeddings.py`: sentence-transformers embeddings with deterministic hashed fallback
- `neuroassist/services/unet_ingest.py`: derived imaging heuristics and persistence into `imaging_reports`

### Database

`neuroassist/db/schema.sql` creates:

- `patients`
- `sessions`
- `patient_context`
- `messages`
- `questions_asked`
- `imaging_reports`
- `diagnostic_reports`
- `knowledge_chunks`

The knowledge base supports `pgvector` when available and falls back to a `DOUBLE PRECISION[]` embedding column if the extension is missing.

## Prerequisites

- Python 3.10+ recommended
- PostgreSQL running locally
- A trained segmentation model at `Tumor Model/best_attention_unet_v2.keras`
- Enough RAM/VRAM for TensorFlow inference

## Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment

This repo now uses a single shared env file:

- root `.env`

There is no longer a separate `neuroassist/.env`.

Important variables in the root `.env`:

- `DATABASE_URL`
- `POSTGRES_ADMIN_DB`
- `AUTO_BOOTSTRAP_DB`
- `LLM_PROVIDER`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_BASE_URL`
- `MODEL_PATH`
- `EMBED_MODEL`
- `PORT`

## Database Bootstrap

Run these from inside `neuroassist/`:

```powershell
cd neuroassist
python -m db.init_db
python -m scripts.seed_knowledge_base
```

`seed_knowledge_base` inserts a small curated WHO 2021 / VASARI / RANO reference set used by the RAG layer.

## Running The App

Run the Flask app from inside `neuroassist/`:

```powershell
cd neuroassist
python app.py
```

Open `http://localhost:5000`.

## Upload Expectations

### NIfTI uploads

- Preferred: a folder containing the four BraTS-style modalities
- Supported modality names: `t1`/`t1w`, `t1ce`/`t1wce`, `t2`/`t2w`, `flair`
- The upload route can also assemble modality sets from zipped NIfTI folders
- A single 4D NIfTI volume is still accepted as a fallback

### DICOM uploads

- Preferred: a folder with modality subfolders such as `t1`, `t1w`, `t1ce`, `t1wce`, `t2`, `t2w`, and `flair`
- Zipped DICOM folders are also supported
- Conversion is handled through `Input_preprocessing/dicom_to_nii_pipeline.py`

## Tests

Run tests from inside `neuroassist/`:

```powershell
cd neuroassist
python -m unittest tests.test_llm tests.test_upload_modalities -v
```

Current tests cover:

- LLM JSON parsing and retry behavior
- final report context composition
- modality alias detection and canonical NIfTI case assembly

## Notes And Limitations

- The report generator is wired to Groq in the current code path.
- Ollama settings remain in `.env` as an optional legacy fallback, but they are not the primary runtime path.
- Large data folders, model weights, uploads, and generated static assets are intentionally ignored by git.
- The preprocessing scripts in `Input_preprocessing/` are utility scripts and are not part of the Flask request path except for DICOM-to-NIfTI conversion.
