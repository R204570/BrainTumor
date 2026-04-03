"""
get_labels.py
--------------
Creates annotated PNG slices from a predicted tumor mask.

This module is import-safe for Flask routes, so it does not execute any
code at import time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so server-side export works.
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Label colors: 0=background(transparent), 1=NCR(red), 2=ED(yellow), 3=ET(cyan)
LABEL_COLORS = {
    1: (255, 80, 80),
    2: (255, 220, 50),
    3: (80, 220, 255),
}
LABEL_NAMES = {
    1: "NCR/NET",
    2: "Edema",
    3: "Enhancing",
}


def annotate_slice(mri_slice: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay colored tumor labels on a grayscale MRI slice.

    Parameters
    ----------
    mri_slice : (H, W) float
    mask_slice: (H, W) uint8 labels 0-3
    alpha     : overlay opacity
    """
    mn, mx = float(mri_slice.min()), float(mri_slice.max())
    if mx > mn:
        gray = ((mri_slice - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        gray = np.zeros_like(mri_slice, dtype=np.uint8)

    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    for label_id, color in LABEL_COLORS.items():
        region = mask_slice == label_id
        if region.any():
            for channel_idx, channel_value in enumerate(color):
                rgb[..., channel_idx] = np.where(
                    region,
                    (1 - alpha) * rgb[..., channel_idx] + alpha * channel_value,
                    rgb[..., channel_idx],
                )

    return np.clip(rgb, 0, 255).astype(np.uint8)


def export_annotated_slices(
    volume: np.ndarray,
    pred_mask: np.ndarray,
    metrics: dict,
    lobe_inv: dict,
    patient_id: str,
    out_dir: str = "/tmp/annotated",
) -> Path:
    """
    Export PNGs for axial slices containing tumor and a summary 3-plane image.
    """
    out_path = Path(out_dir) / patient_id
    out_path.mkdir(parents=True, exist_ok=True)

    # Use FLAIR channel when available; otherwise use first channel.
    flair = volume[..., 3] if volume.ndim == 4 and volume.shape[-1] >= 4 else volume[..., 0]
    centroid = tuple(metrics.get("centroid", (64, 64, 64)))
    cz, cy, cx = (int(centroid[0]), int(centroid[1]), int(centroid[2]))

    tumor_slices = [z for z in range(pred_mask.shape[0]) if np.any(pred_mask[z, :, :] > 0)]

    for z in tumor_slices:
        img = annotate_slice(flair[z, :, :], pred_mask[z, :, :])

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        ax.imshow(img)
        ax.set_title(f"{patient_id} | Axial z={z}", fontsize=8, pad=3)
        ax.axis("off")

        patches = []
        for label_id, color in LABEL_COLORS.items():
            if np.any(pred_mask[z, :, :] == label_id):
                patches.append(
                    mpatches.Patch(
                        color=np.array(color) / 255.0,
                        label=LABEL_NAMES[label_id],
                    )
                )

        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=6, framealpha=0.6)

        plt.tight_layout(pad=0.3)
        plt.savefig(out_path / f"axial_z{z:03d}.png", bbox_inches="tight", dpi=100)
        plt.close(fig)

    # Clamp centroid to valid bounds.
    cz = min(max(cz, 0), pred_mask.shape[0] - 1)
    cy = min(max(cy, 0), pred_mask.shape[1] - 1)
    cx = min(max(cx, 0), pred_mask.shape[2] - 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=120)
    fig.patch.set_facecolor("#111111")

    planes = [
        (annotate_slice(flair[cz, :, :], pred_mask[cz, :, :]), f"Axial z={cz}"),
        (annotate_slice(flair[:, cy, :], pred_mask[:, cy, :]), f"Coronal y={cy}"),
        (annotate_slice(flair[:, :, cx], pred_mask[:, :, cx]), f"Sagittal x={cx}"),
    ]

    for ax, (img, title) in zip(axes, planes):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    ax_lobe = fig.add_axes([0.01, 0.02, 0.28, 0.22])
    lobes = list(lobe_inv.keys())
    pcts = list(lobe_inv.values())
    colors = ["#4CA3DD", "#E86A3A", "#5CC85C", "#B05AC4", "#8A8A8A"][: len(lobes)]
    ax_lobe.barh(lobes, pcts, color=colors, height=0.5)
    ax_lobe.set_xlim(0, 100)
    ax_lobe.set_xlabel("% of tumor", color="white", fontsize=7)
    ax_lobe.tick_params(colors="white", labelsize=7)
    ax_lobe.set_facecolor("#222222")
    for spine in ax_lobe.spines.values():
        spine.set_edgecolor("#555555")

    fig.text(
        0.36,
        0.04,
        "WT: {wt} cm3 | ET: {et} cm3 | NCR: {ncr} cm3 | ED: {ed} cm3 | Diameter: {d} mm".format(
            wt=metrics.get("wt_volume_cm3", 0),
            et=metrics.get("et_volume_cm3", 0),
            ncr=metrics.get("ncr_volume_cm3", 0),
            ed=metrics.get("ed_volume_cm3", 0),
            d=metrics.get("diameter_mm", 0),
        ),
        color="white",
        fontsize=9,
        va="bottom",
    )

    plt.suptitle(f"Brain Tumor Analysis - {patient_id}", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path / "summary_3plane.png", bbox_inches="tight", facecolor="#111111", dpi=120)
    plt.close(fig)

    print(f"[get_labels] Saved {len(tumor_slices)} axial slices and summary to {out_path}")
    return out_path
