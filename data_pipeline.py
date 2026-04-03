"""
glio_pipeline.py
────────────────
Full GBM inference + analysis pipeline.

Functions
---------
predict_patient          – Load, preprocess, and run model inference
compute_tumor_metrics    – Physical volume & geometry from pred mask
build_lobe_atlas         – Gap-free 128³ cerebral lobe atlas
compute_lobe_involvement – Tumor distribution across lobes
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from Load_model import load_model


LOGGER = logging.getLogger(__name__)
MODEL_INPUT_SHAPE = (128, 128, 128)
MODALITIES = ("t1", "t1ce", "t2", "flair")

ALL_LOBE_NAMES = {0: "Other", 1: "Frontal", 2: "Parietal", 3: "Temporal", 4: "Occipital"}
LOBE_NAMES = {1: "Frontal", 2: "Parietal", 3: "Temporal", 4: "Occipital"}


def _find_modality_file(patient_folder: str | Path, patient_id: str, modality: str) -> Path:
    patient_path = Path(patient_folder)
    candidates = [
        patient_path / f"{patient_id}_{modality}.nii.gz",
        patient_path / f"{patient_id}_{modality}.nii",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing modality '{modality}' for patient '{patient_id}'. "
        f"Checked: {', '.join(str(path.name) for path in candidates)}"
    )


def _normalize_zscore(volume: np.ndarray) -> np.ndarray:
    brain_mask = volume > 0
    if not np.any(brain_mask):
        return volume.astype(np.float32, copy=False)

    mean = float(volume[brain_mask].mean())
    std = float(volume[brain_mask].std())
    if std < 1e-8:
        return volume.astype(np.float32, copy=False)

    normalized = (volume - mean) / (std + 1e-8)
    normalized[~brain_mask] = 0.0
    return normalized.astype(np.float32, copy=False)


def _resample_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int] = MODEL_INPUT_SHAPE,
    *,
    order: int = 1,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    zoom_factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume, zoom_factors, order=order).astype(np.float32), zoom_factors


def load_and_preprocess_patient(
    patient_folder: str | Path,
    *,
    return_spacing: bool = False,
):
    """
    Load the 4 BraTS modalities, resample them to 128^3, and z-score normalize.

    Parameters
    ----------
    patient_folder : str | Path
        Folder containing `{patient_id}_{modality}.nii[.gz]` files.
    return_spacing : bool, default False
        When True, also return the resampled voxel spacing tuple in mm.
    """
    patient_path = Path(patient_folder)
    patient_id = patient_path.name

    channels: list[np.ndarray] = []
    voxel_spacing_mm: tuple[float, float, float] | None = None

    for modality in MODALITIES:
        modality_path = _find_modality_file(patient_path, patient_id, modality)
        img = nib.load(str(modality_path))
        volume = img.get_fdata(dtype=np.float32)
        spacing = tuple(float(v) for v in img.header.get_zooms()[:3])

        volume, zoom_factors = _resample_volume(volume, MODEL_INPUT_SHAPE, order=1)
        volume = _normalize_zscore(volume)
        channels.append(volume)

        if voxel_spacing_mm is None:
            voxel_spacing_mm = tuple(sp / zf for sp, zf in zip(spacing, zoom_factors))

    if voxel_spacing_mm is None:
        raise ValueError(f"Could not determine voxel spacing for patient '{patient_id}'.")

    stacked = np.stack(channels, axis=-1).astype(np.float32)
    voxel_vol_mm3 = float(np.prod(voxel_spacing_mm))

    if return_spacing:
        return stacked, voxel_vol_mm3, patient_id, voxel_spacing_mm
    return stacked, voxel_vol_mm3, patient_id

# ── 1. Inference ──────────────────────────────────────────────────────────────

def predict_patient(
    patient_folder,
    *,
    model=None,
    return_spacing: bool = False,
):
    """
    Full inference pipeline for one patient.

    Parameters
    ----------
    patient_folder : str | Path
        Path to the folder containing the patient's MRI modalities.

    Returns
    -------
    volume    : (128,128,128,4) float32  — preprocessed MRI
    pred_mask : (128,128,128)   uint8    — argmax predicted labels
    prob_map  : (128,128,128,4) float32  — raw softmax probabilities
    voxel_vol : float — mm³ per resampled voxel
    pid       : str   — patient ID
    voxel_spacing_mm : tuple[float, float, float] — optional per-axis spacing
    """
    volume, voxel_vol, pid, voxel_spacing_mm = load_and_preprocess_patient(
        patient_folder,
        return_spacing=True,
    )
    model = model or load_model()
    inp = np.expand_dims(volume, axis=0)  # (1,128,128,128,4)

    if hasattr(model, "predict"):
        prob_map = model.predict(inp, verbose=0)[0]
    else:
        output = model(inp)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        prob_map = output.numpy()[0]

    pred_mask = np.argmax(prob_map, axis=-1).astype(np.uint8)
    if return_spacing:
        return volume, pred_mask, prob_map, voxel_vol, pid, voxel_spacing_mm
    return volume, pred_mask, prob_map, voxel_vol, pid


# ── 2. Tumor metrics ──────────────────────────────────────────────────────────

def compute_tumor_metrics(
    pred_mask,
    voxel_vol_mm3,
    voxel_spacing_mm: tuple[float, float, float] | None = None,
):
    """
    Computes physical tumor measurements using true voxel spacing.

    Parameters
    ----------
    pred_mask     : (128,128,128) uint8 — model prediction
    voxel_vol_mm3 : float — mm³ per resampled voxel
    voxel_spacing_mm : tuple[float, float, float] | None
        Optional per-axis spacing `(z, y, x)` in mm. When omitted we fall
        back to an isotropic approximation from voxel volume.

    Returns
    -------
    dict with keys:
        ncr_volume_cm3, ed_volume_cm3, et_volume_cm3, wt_volume_cm3,
        bbox, diameter_mm, centroid
    """
    metrics = {}

    # Per sub-region voxel counts
    ncr_vox = np.sum(pred_mask == 1)   # Necrotic core
    ed_vox  = np.sum(pred_mask == 2)   # Edema
    et_vox  = np.sum(pred_mask == 3)   # Enhancing tumor
    wt_vox  = np.sum(pred_mask > 0)    # Whole tumor

    # Convert to physical volume (mm³ → cm³)
    metrics["ncr_volume_cm3"] = round(ncr_vox * voxel_vol_mm3 / 1000, 2)
    metrics["ed_volume_cm3"]  = round(ed_vox  * voxel_vol_mm3 / 1000, 2)
    metrics["et_volume_cm3"]  = round(et_vox  * voxel_vol_mm3 / 1000, 2)
    metrics["wt_volume_cm3"]  = round(wt_vox  * voxel_vol_mm3 / 1000, 2)

    # 3-D bounding box & max diameter
    coords = np.where(pred_mask > 0)
    if len(coords[0]) > 0:
        z0, z1 = coords[0].min(), coords[0].max()
        y0, y1 = coords[1].min(), coords[1].max()
        x0, x1 = coords[2].min(), coords[2].max()
        metrics["bbox"] = (z0, z1, y0, y1, x0, x1)

        if voxel_spacing_mm is None:
            approx_spacing_mm = voxel_vol_mm3 ** (1 / 3)
            voxel_spacing_mm = (approx_spacing_mm, approx_spacing_mm, approx_spacing_mm)
            LOGGER.warning(
                "compute_tumor_metrics() received only voxel volume; "
                "diameter_mm is using an isotropic spacing approximation."
            )

        z_mm = (z1 - z0) * float(voxel_spacing_mm[0])
        y_mm = (y1 - y0) * float(voxel_spacing_mm[1])
        x_mm = (x1 - x0) * float(voxel_spacing_mm[2])
        metrics["diameter_mm"] = round(max(z_mm, y_mm, x_mm), 1)
        metrics["centroid"] = (
            int((z0 + z1) / 2),
            int((y0 + y1) / 2),
            int((x0 + x1) / 2),
        )
    else:
        metrics["bbox"]        = None
        metrics["diameter_mm"] = 0
        metrics["centroid"]    = (64, 64, 64)

    return metrics


# ── 3. Lobe atlas ─────────────────────────────────────────────────────────────

def build_lobe_atlas(shape=(128, 128, 128)):
    """
    Builds a simple MNI-space approximation of the 4 cerebral lobes
    at 128³ resolution.

    Axes in BraTS resampled space
    ──────────────────────────────
        axis 0 (z) = inferior → superior
        axis 1 (y) = posterior → anterior
        axis 2 (x) = right → left

    Labels
    ──────
        0 = Other (brainstem, cerebellum, background)
        1 = Frontal lobe
        2 = Parietal lobe
        3 = Temporal lobe
        4 = Occipital lobe

    Note: initialised as all-1 (Frontal) so every voxel belongs to a
    lobe from the start — prevents silent "Other" gaps swallowing
    tumor voxels and breaking the 100 % percentage sum.

    Parameters
    ----------
    shape : tuple, default (128, 128, 128)

    Returns
    -------
    atlas : uint8 ndarray of the given shape
    """
    D, H, W = shape

    # Fill everything as Frontal first (prevents unlabelled gaps)
    atlas = np.ones(shape, dtype=np.uint8)

    # Proportional boundaries
    z_mid  = int(D * 0.50)
    z_sup  = int(D * 0.72)

    y_front_end = int(H * 0.72)
    y_par_start = int(H * 0.38)
    y_occ_start = int(H * 0.18)

    z_temp_max = int(D * 0.62)
    y_temp_end = int(H * 0.62)

    # Parietal: superior mid-posterior
    atlas[z_sup:,      y_occ_start:y_par_start, :] = 2
    atlas[z_mid:z_sup, int(H * 0.25):y_par_start, :] = 2

    # Occipital: posterior pole (first pass)
    atlas[:, :y_occ_start, :] = 4

    # Temporal: inferior lateral (overrides inferior-anterior block)
    atlas[:z_temp_max, :y_temp_end, :] = 3

    # Re-assert Occipital at posterior pole (highest priority)
    atlas[:, :y_occ_start, :] = 4

    return atlas


# ── 4. Lobe involvement ───────────────────────────────────────────────────────

def compute_lobe_involvement(
    pred_mask,
    lobe_atlas,
    *,
    include_other: bool = False,
):
    """
    For each brain lobe, compute what percentage of the whole tumor
    falls within that lobe.

    Parameters
    ----------
    pred_mask  : (128,128,128) uint8 — model prediction
    lobe_atlas : (128,128,128) uint8 — output of build_lobe_atlas()
    include_other : bool, default False
        Include label 0 ("Other") only when explicitly requested. The atlas
        is intended to be gap-free, so "Other" is hidden from normal reports.

    Returns
    -------
    dict : lobe name → float percentage (named lobes only by default)
    """
    tumor_mask = pred_mask > 0
    tumor_total = int(tumor_mask.sum())

    if tumor_total == 0:
        empty = {name: 0.0 for name in LOBE_NAMES.values()}
        if include_other:
            empty["Other"] = 0.0
        return empty

    involvement = {}
    for lobe_id, lobe_name in LOBE_NAMES.items():
        lobe_region = lobe_atlas == lobe_id
        overlap = int(np.logical_and(tumor_mask, lobe_region).sum())
        involvement[lobe_name] = round(100.0 * overlap / tumor_total, 1)

    other_overlap = int(np.logical_and(tumor_mask, lobe_atlas == 0).sum())
    other_pct = round(100.0 * other_overlap / tumor_total, 1)
    total_pct = sum(involvement.values()) + other_pct
    if not 99.0 <= total_pct <= 101.0:
        LOGGER.warning(
            "Lobe percentages sum to %.1f%%; atlas may contain unlabeled voxels. "
            "Computed Other overlap: %.1f%%.",
            total_pct,
            other_pct,
        )

    named = dict(sorted(involvement.items(), key=lambda item: item[1], reverse=True))
    if include_other and other_pct > 0.0:
        named["Other"] = other_pct
    return named


def format_pipeline_context(pipeline_output: dict, *, include_other: bool = False) -> str:
    """
    Convert pipeline output into a compact human-readable block for LLM prompts.
    """
    wt_volume = float(pipeline_output.get("wt_volume_cm3") or 0.0)
    centroid = pipeline_output.get("centroid") or (64, 64, 64)
    bbox = pipeline_output.get("bbox")

    lobe_involvement = dict(pipeline_output.get("lobe_involvement") or {})
    if not include_other:
        lobe_involvement.pop("Other", None)

    if lobe_involvement:
        lobe_lines = [
            f"  - {name}: {float(pct):.1f}%"
            for name, pct in sorted(
                lobe_involvement.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
    else:
        lobe_lines = ["  - No lobar involvement detected"]

    bbox_text = (
        f"z={bbox[0]}..{bbox[1]}, y={bbox[2]}..{bbox[3]}, x={bbox[4]}..{bbox[5]}"
        if bbox
        else "None"
    )

    return (
        "Imaging Pipeline Summary\n"
        f"- Tumor detected: {'Yes' if wt_volume > 0 else 'No'}\n"
        f"- Whole tumor volume: {wt_volume:.2f} cm3\n"
        f"- Necrotic core volume: {float(pipeline_output.get('ncr_volume_cm3') or 0.0):.2f} cm3\n"
        f"- Edema volume: {float(pipeline_output.get('ed_volume_cm3') or 0.0):.2f} cm3\n"
        f"- Enhancing tumor volume: {float(pipeline_output.get('et_volume_cm3') or 0.0):.2f} cm3\n"
        f"- Maximum diameter: {float(pipeline_output.get('diameter_mm') or 0.0):.1f} mm\n"
        f"- Centroid (z, y, x): ({int(centroid[0])}, {int(centroid[1])}, {int(centroid[2])})\n"
        f"- Bounding box: {bbox_text}\n"
        "- Lobe involvement:\n"
        + "\n".join(lobe_lines)
    )


# ── Quick self-test (run: python glio_pipeline.py) ────────────────────────────

if __name__ == "__main__":
    print("Building lobe atlas …")
    atlas = build_lobe_atlas()

    total = 128 ** 3
    print(f"Total voxels : {total:,}\n")
    for k, v in ALL_LOBE_NAMES.items():
        count = int(np.sum(atlas == k))
        print(f"  {v:12s}: {count:>8,}  ({100*count/total:5.1f}%)")

    other = int(np.sum(atlas == 0))
    if other:
        LOGGER.warning("%s unlabeled voxels remain in the atlas.", other)
    else:
        print("\nAtlas is gap-free. All voxels are assigned to named lobes.")
