"""
routes/upload.py - POST /upload/scan

Uploads MRI scans, optionally converts DICOM to NIfTI, runs model inference,
exports annotated slices, and persists imaging metrics.
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from db.connection import execute_query
from services.unet_ingest import compute_derived_metrics, ingest_unet_output

upload_bp = Blueprint("upload", __name__)

# Existing model and preprocessing scripts live one level up from this Flask app.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")
MODALITY_ALIASES = {
    "t1": ("t1", "t1w"),
    "t1ce": ("t1ce", "t1wce"),
    "t2": ("t2", "t2w"),
    "flair": ("flair",),
}
MODALITY_HINTS = {alias for aliases in MODALITY_ALIASES.values() for alias in aliases}
MODALITY_TOKEN_ORDER = ("t1wce", "t1ce", "flair", "t2w", "t2", "t1w", "t1")


@upload_bp.route("/upload/scan", methods=["POST"])
def upload_scan():
    """
    POST /upload/scan

    Form fields:
      session_id: UUID
      scan_type : 'nii' | 'dicom'
      files[]   : folder upload (preferred)
      file      : legacy single-file upload (fallback)
    """
    session_id = (request.form.get("session_id") or "").strip()
    scan_type = (request.form.get("scan_type") or "nii").strip().lower()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if scan_type not in {"nii", "dicom"}:
        return jsonify({"error": "scan_type must be 'nii' or 'dicom'"}), 400

    sess_row = execute_query(
        "SELECT patient_id FROM sessions WHERE id = %s",
        (session_id,),
        fetch="one",
    )
    if not sess_row:
        return jsonify({"error": "Session not found"}), 404

    patient_id = str(sess_row["patient_id"])

    uploaded_files = _collect_uploaded_files()
    if not uploaded_files:
        return jsonify({"error": "No files uploaded"}), 400

    upload_dir = Path(current_app.config["UPLOAD_FOLDER"]) / session_id
    raw_input_dir = upload_dir / "raw_input"
    raw_input_dir.mkdir(parents=True, exist_ok=True)

    # Replace previous raw upload payload for this session.
    for old in raw_input_dir.iterdir():
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()

    saved_paths = []
    saved_rel_paths = []
    for idx, uploaded in enumerate(uploaded_files, start=1):
        rel_path = _safe_relative_upload_path(uploaded.filename, fallback=f"upload_{idx}")
        save_path = raw_input_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(str(save_path))
        saved_paths.append(save_path)
        saved_rel_paths.append(rel_path)

    original_name = _build_upload_label(saved_rel_paths)

    try:
        if scan_type == "nii":
            nii_path = _prepare_nii_input(saved_paths, upload_dir)
        else:
            nii_path = _convert_dicom_to_nii(saved_paths, raw_input_dir, upload_dir)

        if not nii_path.exists():
            return jsonify({"error": "NIfTI file could not be prepared"}), 500

        unet_output = _run_inference(nii_path, patient_id)

        derived = compute_derived_metrics(unet_output)

        imaging_report_id = ingest_unet_output(
            patient_id=patient_id,
            session_id=session_id,
            unet_output=unet_output,
            scan_filename=original_name,
            scan_format=scan_type,
            scan_date=date.today(),
            annotated_dir=unet_output.get("annotated_dir", ""),
        )

        return jsonify(
            {
                "imaging_report_id": imaging_report_id,
                "metrics": {
                    "wt_volume_cm3": unet_output.get("wt_volume_cm3"),
                    "ncr_volume_cm3": unet_output.get("ncr_volume_cm3"),
                    "ed_volume_cm3": unet_output.get("ed_volume_cm3"),
                    "et_volume_cm3": unet_output.get("et_volume_cm3"),
                    "diameter_mm": unet_output.get("diameter_mm"),
                    "lobe_involvement": unet_output.get("lobe_involvement", {}),
                    "derived": {
                        "necrosis_ratio": derived.get("necrosis_ratio"),
                        "enhancement_ratio": derived.get("enhancement_ratio"),
                        "edema_ratio": derived.get("edema_ratio"),
                        "midline_shift_mm": derived.get("midline_shift_mm"),
                    },
                },
                "annotated_dir": unet_output.get("annotated_dir", ""),
            }
        ), 201

    except Exception as exc:
        current_app.logger.exception("Upload/inference error")
        return jsonify({"error": str(exc)}), 500


def _collect_uploaded_files():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    files += [f for f in request.files.getlist("files[]") if f and f.filename]
    if files:
        return files

    # Backward compatibility for old frontend using single "file" key.
    legacy = request.files.get("file")
    if legacy and legacy.filename:
        return [legacy]

    return []


def _safe_relative_upload_path(filename: str, fallback: str) -> Path:
    normalized = (filename or "").replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return Path(fallback)

    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Invalid upload path: {filename}")

    safe_parts = []
    for part in pure.parts:
        if part in {"", "."}:
            continue
        safe = secure_filename(part)
        if safe:
            safe_parts.append(safe)

    if not safe_parts:
        safe_parts = [fallback]

    return Path(*safe_parts)


def _build_upload_label(rel_paths: list[Path]) -> str:
    if not rel_paths:
        return "uploaded_scan"
    if len(rel_paths) == 1:
        return rel_paths[0].name

    roots = {p.parts[0] for p in rel_paths if p.parts}
    if len(roots) == 1:
        root = next(iter(roots))
        return f"{root} ({len(rel_paths)} files)"

    return f"folder_upload ({len(rel_paths)} files)"


def _is_nifti_file(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _prepare_nii_input(uploaded_paths: list[Path], work_dir: Path) -> Path:
    nii_files = [p for p in uploaded_paths if _is_nifti_file(p)]
    if not nii_files:
        zip_files = [p for p in uploaded_paths if p.suffix.lower() == ".zip"]
        if zip_files:
            extract_root = work_dir / "nii_input"
            _reset_dir(extract_root)
            for archive_path in zip_files:
                target = extract_root / archive_path.stem
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path, "r") as archive:
                    archive.extractall(str(target))
            nii_files = sorted([*extract_root.rglob("*.nii"), *extract_root.rglob("*.nii.gz")])

    if not nii_files:
        raise FileNotFoundError("No .nii or .nii.gz files found in uploaded folder")

    return _prepare_nifti_case_input(nii_files, work_dir)


def _convert_dicom_to_nii(uploaded_paths: list[Path], raw_input_root: Path, work_dir: Path) -> Path:
    """
    Convert DICOM upload to NIfTI using Input_preprocessing/dicom_to_nii_pipeline.py.

    The converter expects a root folder containing modality subfolders.
    This helper supports:
      1) direct folder upload of .dcm files with modality subfolders
      2) zipped DICOM folders (legacy)
    """
    from Input_preprocessing.dicom_to_nii_pipeline import run_reverse_pipeline

    extract_root = work_dir / "dicom_input"
    nii_output = work_dir / "nii_output"
    _reset_dir(extract_root)
    _reset_dir(nii_output)

    zip_inputs = [p for p in uploaded_paths if p.suffix.lower() == ".zip"]
    dcm_inputs = [p for p in uploaded_paths if p.suffix.lower() == ".dcm"]

    if not zip_inputs and not dcm_inputs:
        raise FileNotFoundError("For DICOM upload, provide .dcm files or a .zip folder")

    candidate_roots = []

    if dcm_inputs:
        candidate_roots.append(raw_input_root)

    for archive_path in zip_inputs:
        target = extract_root / archive_path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(str(target))
        candidate_roots.append(target)

    if not candidate_roots:
        candidate_roots.append(raw_input_root)

    dicom_root = None
    for candidate_root in candidate_roots:
        dicom_root = _find_modality_root(candidate_root)
        if dicom_root is not None:
            break

    if dicom_root is None:
        # Fallback: if there are raw DICOM files without modality folders,
        # collect them under flair/ so the converter can still process them.
        loose_dicoms = []
        for candidate_root in candidate_roots:
            loose_dicoms.extend(candidate_root.rglob("*.dcm"))

        if loose_dicoms:
            dicom_root = extract_root / "fallback_series"
            flair_dir = dicom_root / "flair"
            flair_dir.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(loose_dicoms, start=1):
                dst = flair_dir / f"slice_{idx:04d}.dcm"
                shutil.copy2(src, dst)

    if dicom_root is None:
        raise FileNotFoundError(
            "Could not find modality subfolders (t1/t1w, t2/t2w, t1wce/t1ce, flair) in DICOM upload"
        )

    run_reverse_pipeline(str(dicom_root), str(nii_output))

    nii_files = sorted([*nii_output.glob("*.nii"), *nii_output.glob("*.nii.gz")])
    if not nii_files:
        raise FileNotFoundError("DICOM conversion produced no NIfTI files")

    return _prepare_nifti_case_input(nii_files, work_dir)


def _prepare_nifti_case_input(nii_files: list[Path], work_dir: Path) -> Path:
    filtered = [p for p in nii_files if "seg" not in p.name.lower()]
    candidates = filtered or nii_files
    modality_files = _match_modality_files(candidates)

    if len(modality_files) == len(CANONICAL_MODALITIES):
        return _materialize_canonical_case(modality_files, work_dir)

    if len(candidates) == 1:
        # Backward compatibility: permit a single pre-stacked 4D NIfTI volume.
        return candidates[0]

    missing = [mod for mod in CANONICAL_MODALITIES if mod not in modality_files]
    found = ", ".join(sorted(modality_files)) or "none"
    raise FileNotFoundError(
        "Could not assemble the full BraTS modality set from the uploaded NIfTI files. "
        f"Found: {found}. Missing: {', '.join(missing)}. "
        "Accepted aliases: t1/t1w, t1ce/t1wce, t2/t2w, flair."
    )


def _match_modality_files(nii_files: list[Path]) -> dict[str, Path]:
    matched: dict[str, tuple[int, Path]] = {}

    for path in sorted(nii_files):
        token = _detect_modality_token(path.name)
        if not token:
            continue

        canonical = _canonicalize_modality(token)
        rank = _modality_alias_rank(token, canonical)
        current = matched.get(canonical)
        if current is None or rank < current[0]:
            matched[canonical] = (rank, path)

    return {canonical: path for canonical, (_, path) in matched.items()}


def _detect_modality_token(filename: str) -> str | None:
    stem = _strip_nifti_suffix(filename)
    normalized = re.sub(r"[\s.\-]+", "_", stem.lower())

    for token in MODALITY_TOKEN_ORDER:
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", normalized):
            return token
    return None


def _strip_nifti_suffix(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".nii.gz"):
        return filename[:-7]
    if lowered.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def _canonicalize_modality(token: str) -> str:
    for canonical, aliases in MODALITY_ALIASES.items():
        if token in aliases:
            return canonical
    raise ValueError(f"Unknown modality token: {token}")


def _modality_alias_rank(token: str, canonical: str) -> int:
    aliases = MODALITY_ALIASES[canonical]
    return aliases.index(token) if token in aliases else len(aliases)


def _materialize_canonical_case(modality_files: dict[str, Path], work_dir: Path) -> Path:
    case_dir = work_dir / "nii_case" / "uploaded_case"
    _reset_dir(case_dir)

    for canonical in CANONICAL_MODALITIES:
        source = modality_files[canonical]
        suffix = ".nii.gz" if source.name.lower().endswith(".nii.gz") else ".nii"
        target = case_dir / f"uploaded_case_{canonical}{suffix}"
        shutil.copy2(source, target)

    return case_dir


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _find_modality_root(root: Path) -> Path | None:
    """
    Return the first directory that directly contains modality subfolders.
    """
    if _looks_like_modality_root(root):
        return root

    for candidate in root.rglob("*"):
        if candidate.is_dir() and _looks_like_modality_root(candidate):
            return candidate

    return None


def _looks_like_modality_root(path: Path) -> bool:
    child_dirs = {p.name.lower() for p in path.iterdir() if p.is_dir()}
    normalized = {"t1wce" if name == "t1ce" else name for name in child_dirs}
    return len(normalized.intersection(MODALITY_HINTS)) > 0


def _run_inference(nii_path: Path, patient_id: str) -> dict:
    """
    Load model, run inference on prepared NIfTI, compute metrics, and export labels.
    """
    from Load_model import load_model
    from data_pipeline import (
        build_lobe_atlas,
        compute_lobe_involvement,
        compute_tumor_metrics,
        predict_patient,
    )
    from get_labels import export_annotated_slices

    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom

    model = load_model(current_app.config["MODEL_PATH"])

    if nii_path.is_dir():
        data, pred_mask, prob_map, voxel_vol_mm3, _case_id, voxel_spacing_mm = predict_patient(
            nii_path,
            model=model,
            return_spacing=True,
        )
    else:
        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)
        zooms = img.header.get_zooms()[:3]

        target_shape = (128, 128, 128)
        if data.shape[:3] != target_shape:
            factors = [t / s for t, s in zip(target_shape, data.shape[:3])]
            if data.ndim == 4:
                data = np.stack([zoom(data[..., c], factors, order=1) for c in range(data.shape[3])], axis=-1)
            else:
                data = zoom(data, factors, order=1)

        if data.ndim == 3:
            # Single-modality fallback for legacy uploads; preferred path is 4 separate modalities.
            data = np.stack([data] * 4, axis=-1)
        elif data.ndim == 4 and data.shape[-1] > 4:
            data = data[..., :4]

        for c in range(data.shape[-1]):
            channel = data[..., c]
            brain_mask = channel > 0
            if np.any(brain_mask):
                mean = float(channel[brain_mask].mean())
                std = float(channel[brain_mask].std())
                if std >= 1e-8:
                    channel = (channel - mean) / (std + 1e-8)
                    channel[~brain_mask] = 0.0
            data[..., c] = channel

        original_shape = img.shape[:3]
        zoom_factors = [128 / s for s in original_shape]
        voxel_spacing_mm = tuple(
            float(spacing / factor)
            for spacing, factor in zip(zooms, zoom_factors)
        )
        voxel_vol_mm3 = float(np.prod(voxel_spacing_mm))

        inp = np.expand_dims(data, axis=0)

        if hasattr(model, "predict"):
            prob_map = model.predict(inp, verbose=0)[0]
        else:
            # SavedModel callable fallback.
            output = model(inp)
            if isinstance(output, dict):
                output = next(iter(output.values()))
            prob_map = output.numpy()[0]

        pred_mask = np.argmax(prob_map, axis=-1).astype(np.uint8)

    metrics = compute_tumor_metrics(pred_mask, voxel_vol_mm3, voxel_spacing_mm)
    atlas = build_lobe_atlas()
    lobe_inv = compute_lobe_involvement(pred_mask, atlas)

    annotated_base = Path(current_app.config["ANNOTATED_FOLDER"])
    out_path = export_annotated_slices(
        volume=data,
        pred_mask=pred_mask,
        metrics=metrics,
        lobe_inv=lobe_inv,
        patient_id=patient_id,
        out_dir=str(annotated_base),
    )

    static_dir = Path(current_app.static_folder)
    try:
        rel = Path(out_path).relative_to(static_dir)
        annotated_web = "/static/" + str(rel).replace("\\", "/")
    except ValueError:
        annotated_web = str(out_path)

    return {
        **metrics,
        "lobe_involvement": lobe_inv,
        "annotated_dir": annotated_web,
    }
