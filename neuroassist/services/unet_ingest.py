"""
services/unet_ingest.py
───────────────────────
Takes the structured output produced by data_pipeline.py + get_labels.py,
computes derived imaging metrics, and writes a row to the imaging_reports table.

The caller (upload route) is responsible for:
  1. Running the DICOM→NIfTI conversion if needed
  2. Running Load_model.py + data_pipeline functions
  3. Calling ingest_unet_output() with the assembled dict
"""

import math
from datetime import date
from db.connection import execute_query


# ─── Threshold for binary lobe involvement (>10 % of tumor in that lobe) ─────
LOBE_INVOLVEMENT_THRESHOLD = 10.0


def compute_derived_metrics(unet_output: dict) -> dict:
    """
    Compute ratio-based derived metrics from raw U-Net volumes.

    Parameters
    ----------
    unet_output : dict containing keys
        wt_volume_cm3, ncr_volume_cm3, ed_volume_cm3, et_volume_cm3,
        diameter_mm, centroid (tuple), lobe_involvement (dict)

    Returns
    -------
    dict with additional derived keys:
        necrosis_ratio, enhancement_ratio, edema_ratio,
        midline_shift_mm, surface_to_volume
    """
    wt  = unet_output.get("wt_volume_cm3", 0) or 0
    ncr = unet_output.get("ncr_volume_cm3", 0) or 0
    ed  = unet_output.get("ed_volume_cm3", 0) or 0
    et  = unet_output.get("et_volume_cm3", 0) or 0

    derived = {}

    # Ratios relative to whole tumor (safe division)
    derived["necrosis_ratio"]    = round(ncr / wt, 4) if wt > 0 else 0.0
    derived["enhancement_ratio"] = round(et  / wt, 4) if wt > 0 else 0.0
    derived["edema_ratio"]       = round(ed  / wt, 4) if wt > 0 else 0.0

    # Midline shift estimate: rough heuristic from centroid X offset from centre
    # BraTS 128³ space — voxel ≈ 1 mm. Centre = 64.
    centroid = unet_output.get("centroid", (64, 64, 64))
    cx = centroid[2] if isinstance(centroid, (tuple, list)) else 64
    derived["midline_shift_mm"] = round(abs(cx - 64) * 1.0, 2)  # 1 mm/voxel approx.

    # Surface-to-volume ratio (sphere equivalent): S = 4π r²; V = wt in cm³
    # r = (3V/4π)^(1/3)
    if wt > 0:
        r_cm = (3 * wt / (4 * math.pi)) ** (1 / 3)
        surface = 4 * math.pi * r_cm ** 2
        derived["surface_to_volume"] = round(surface / wt, 4)
    else:
        derived["surface_to_volume"] = 0.0

    return derived


def ingest_unet_output(
    patient_id: str,
    session_id: str,
    unet_output: dict,
    scan_filename: str = "",
    scan_format: str = "nii",
    scan_date: date | None = None,
    annotated_dir: str = "",
) -> str:
    """
    Compute derived metrics from U-Net output and persist to imaging_reports.

    Parameters
    ----------
    patient_id     : UUID string
    session_id     : UUID string
    unet_output    : dict from data_pipeline.compute_tumor_metrics() +
                     compute_lobe_involvement()
    scan_filename  : original uploaded file name
    scan_format    : 'nii' | 'dicom'
    scan_date      : date of the scan (default: today)
    annotated_dir  : path to folder containing annotated PNG slices

    Returns
    -------
    imaging_report_id : UUID of the new imaging_reports row
    """
    if scan_date is None:
        scan_date = date.today()

    derived = compute_derived_metrics(unet_output)

    centroid = unet_output.get("centroid", (64, 64, 64))
    if isinstance(centroid, (tuple, list)) and len(centroid) == 3:
        cx, cy, cz = int(centroid[2]), int(centroid[1]), int(centroid[0])
    else:
        cx = cy = cz = 64

    # Lobe involvement dict — e.g. {'Frontal': 55.3, 'Temporal': 32.1, ...}
    lobe_inv = unet_output.get("lobe_involvement", {})

    def lobe_pct(name: str) -> float:
        return lobe_inv.get(name, 0.0)

    def lobe_bool(name: str) -> bool:
        return lobe_pct(name) >= LOBE_INVOLVEMENT_THRESHOLD

    sql = """
        INSERT INTO imaging_reports (
            patient_id, session_id,
            scan_filename, scan_format, scan_date,
            wt_volume_cm3, ncr_volume_cm3, ed_volume_cm3, et_volume_cm3,
            diameter_mm,
            centroid_x, centroid_y, centroid_z,
            lobe_frontal, lobe_temporal, lobe_parietal, lobe_occipital, lobe_other,
            lobe_pct_frontal, lobe_pct_temporal, lobe_pct_parietal,
            lobe_pct_occipital, lobe_pct_other,
            necrosis_ratio, enhancement_ratio, edema_ratio,
            midline_shift_mm, surface_to_volume,
            annotated_dir
        )
        VALUES (
            %(patient_id)s, %(session_id)s,
            %(scan_filename)s, %(scan_format)s, %(scan_date)s,
            %(wt_volume_cm3)s, %(ncr_volume_cm3)s, %(ed_volume_cm3)s, %(et_volume_cm3)s,
            %(diameter_mm)s,
            %(centroid_x)s, %(centroid_y)s, %(centroid_z)s,
            %(lobe_frontal)s, %(lobe_temporal)s, %(lobe_parietal)s,
            %(lobe_occipital)s, %(lobe_other)s,
            %(lobe_pct_frontal)s, %(lobe_pct_temporal)s, %(lobe_pct_parietal)s,
            %(lobe_pct_occipital)s, %(lobe_pct_other)s,
            %(necrosis_ratio)s, %(enhancement_ratio)s, %(edema_ratio)s,
            %(midline_shift_mm)s, %(surface_to_volume)s,
            %(annotated_dir)s
        )
        RETURNING id
    """

    params = {
        "patient_id":      patient_id,
        "session_id":      session_id,
        "scan_filename":   scan_filename,
        "scan_format":     scan_format,
        "scan_date":       scan_date,
        "wt_volume_cm3":   unet_output.get("wt_volume_cm3", 0),
        "ncr_volume_cm3":  unet_output.get("ncr_volume_cm3", 0),
        "ed_volume_cm3":   unet_output.get("ed_volume_cm3", 0),
        "et_volume_cm3":   unet_output.get("et_volume_cm3", 0),
        "diameter_mm":     unet_output.get("diameter_mm", 0),
        "centroid_x":      cx,
        "centroid_y":      cy,
        "centroid_z":      cz,
        "lobe_frontal":    lobe_bool("Frontal"),
        "lobe_temporal":   lobe_bool("Temporal"),
        "lobe_parietal":   lobe_bool("Parietal"),
        "lobe_occipital":  lobe_bool("Occipital"),
        "lobe_other":      lobe_bool("Other"),
        "lobe_pct_frontal":   lobe_pct("Frontal"),
        "lobe_pct_temporal":  lobe_pct("Temporal"),
        "lobe_pct_parietal":  lobe_pct("Parietal"),
        "lobe_pct_occipital": lobe_pct("Occipital"),
        "lobe_pct_other":     lobe_pct("Other"),
        "necrosis_ratio":    derived["necrosis_ratio"],
        "enhancement_ratio": derived["enhancement_ratio"],
        "edema_ratio":       derived["edema_ratio"],
        "midline_shift_mm":  derived["midline_shift_mm"],
        "surface_to_volume": derived["surface_to_volume"],
        "annotated_dir":     annotated_dir,
    }

    row = execute_query(sql, params, fetch="one")
    imaging_report_id = str(row["id"])
    print(f"[unet_ingest] imaging_report created: {imaging_report_id}")
    return imaging_report_id


def get_unet_output_dict(imaging_report_id: str) -> dict:
    """
    Load a stored imaging report row and return it as the unet_outputs
    dict structure expected by context_builder.py.

    Returns
    -------
    dict matching the input_context.unet_outputs schema
    """
    row = execute_query(
        "SELECT * FROM imaging_reports WHERE id = %s",
        (imaging_report_id,),
        fetch="one",
    )
    if not row:
        return {}

    return {
        "wt_volume_cm3":  float(row.get("wt_volume_cm3") or 0),
        "ncr_volume_cm3": float(row.get("ncr_volume_cm3") or 0),
        "ed_volume_cm3":  float(row.get("ed_volume_cm3") or 0),
        "et_volume_cm3":  float(row.get("et_volume_cm3") or 0),
        "diameter_mm":    float(row.get("diameter_mm") or 0),
        "lobe_frontal":   bool(row.get("lobe_frontal")),
        "lobe_temporal":  bool(row.get("lobe_temporal")),
        "lobe_parietal":  bool(row.get("lobe_parietal")),
        "lobe_occipital": bool(row.get("lobe_occipital")),
        "lobe_other":     bool(row.get("lobe_other")),
        "derived": {
            "necrosis_ratio":    float(row.get("necrosis_ratio") or 0),
            "enhancement_ratio": float(row.get("enhancement_ratio") or 0),
            "edema_ratio":       float(row.get("edema_ratio") or 0),
            "midline_shift_mm":  float(row.get("midline_shift_mm") or 0),
        },
        "annotated_dir": row.get("annotated_dir", ""),
    }
