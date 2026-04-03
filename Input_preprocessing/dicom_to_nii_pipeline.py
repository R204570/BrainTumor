"""
DICOM → NIfTI Reverse Pipeline  (Kaggle / Jupyter compatible)
=============================================================
Reads .dcm slices from modality sub-folders and converts each
to a single .nii.gz file.

Expected input structure:
    patient1/
    ├── t1/         slice_0001.dcm … slice_NNNN.dcm
    ├── t1wce/      slice_0001.dcm … slice_NNNN.dcm
    ├── t2/         slice_0001.dcm … slice_NNNN.dcm
    └── flair/      slice_0001.dcm … slice_NNNN.dcm

Output (all .nii.gz files in OUTPUT_FOLDER):
    patient1_t1.nii.gz
    patient1_t1wce.nii.gz
    patient1_t2.nii.gz
    patient1_flair.nii.gz

HOW TO USE:
    1. Set ROOT_FOLDER  → the patient folder containing modality sub-folders
    2. Set OUTPUT_FOLDER → where .nii.gz files will be saved
    3. Run the cell

Requirements:
    pip install pydicom nibabel numpy
"""

import os
import re
import sys
import warnings
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURE THESE PATHS BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

ROOT_FOLDER   = ""   # folder with t1/ t1wce/ t2/ flair/ inside
OUTPUT_FOLDER = ""            # .nii.gz files saved here
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    sys.exit("pydicom not found.  Run:  !pip install pydicom")

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not found.  Run:  !pip install nibabel")


# ── recognised modality folder names ─────────────────────────────────────────
KNOWN_MODALITIES = {"t1", "t1wce", "t1ce", "t2", "flair", "t1w", "t2w"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_dicom_series(folder: str):
    """
    Load all .dcm files from *folder*, sort by InstanceNumber / SliceLocation,
    and return (pixel_volume, affine, modality_name).

    pixel_volume : np.ndarray  shape (rows, cols, n_slices)  float32
    affine       : np.ndarray  shape (4, 4)
    """
    dcm_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ])

    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in: {folder}")

    # Read all slices
    slices = []
    for fname in dcm_files:
        fpath = os.path.join(folder, fname)
        try:
            ds = pydicom.dcmread(fpath, force=True)
            slices.append(ds)
        except (InvalidDicomError, Exception) as e:
            warnings.warn(f"Skipping unreadable file {fname}: {e}")

    if not slices:
        raise RuntimeError(f"Could not read any DICOM files in: {folder}")

    # Sort slices: prefer InstanceNumber, fallback to SliceLocation, then filename
    def sort_key(ds):
        try:
            return float(ds.InstanceNumber)
        except Exception:
            pass
        try:
            return float(ds.SliceLocation)
        except Exception:
            pass
        return 0.0

    slices.sort(key=sort_key)

    # Stack pixel arrays  →  (rows, cols, n_slices)
    pixel_arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)

        # Apply rescale if present
        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept

        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=-1)   # (rows, cols, n_slices)

    # ── build affine from DICOM geometry ─────────────────────────────────
    affine = build_affine(slices)

    # Modality string from DICOM tag (may be empty)
    modality_str = getattr(slices[0], "Modality", "MR")

    return volume, affine, modality_str


def build_affine(slices) -> np.ndarray:
    """
    Reconstruct a 4×4 RAS affine from DICOM ImageOrientationPatient,
    ImagePositionPatient, PixelSpacing, and SliceThickness.
    Falls back to identity if tags are missing.
    """
    try:
        ds0 = slices[0]

        orient = [float(v) for v in ds0.ImageOrientationPatient]
        F = np.array(orient).reshape(2, 3)          # row / col cosines
        row_cosine = F[0]
        col_cosine = F[1]
        normal     = np.cross(row_cosine, col_cosine)

        pos0 = np.array([float(v) for v in ds0.ImagePositionPatient])

        pixel_spacing = ds0.PixelSpacing
        dr = float(pixel_spacing[1])   # col spacing  (x)
        dc = float(pixel_spacing[0])   # row spacing  (y)

        # Slice spacing: use gap between first two ImagePositionPatient if available
        if len(slices) > 1:
            try:
                pos1 = np.array([float(v) for v in slices[1].ImagePositionPatient])
                dz = np.linalg.norm(pos1 - pos0)
            except Exception:
                dz = float(getattr(ds0, "SliceThickness", 1.0))
        else:
            dz = float(getattr(ds0, "SliceThickness", 1.0))

        affine = np.eye(4)
        affine[:3, 0] = row_cosine * dr
        affine[:3, 1] = col_cosine * dc
        affine[:3, 2] = normal     * dz
        affine[:3, 3] = pos0

        return affine

    except Exception as e:
        warnings.warn(f"Could not build affine from DICOM tags ({e}). Using identity.")
        return np.eye(4)


def dcm_folder_to_nii(
    dcm_folder:    str,
    output_path:   str,
    modality_label: str,
    patient_label:  str,
):
    """Convert one DICOM series folder to a NIfTI file."""
    print(f"  [{modality_label.upper():5s}]  reading {len(os.listdir(dcm_folder))} files from {os.path.basename(dcm_folder)}/")

    volume, affine, _ = load_dicom_series(dcm_folder)

    # nibabel convention: (i, j, k) → transpose to (cols, rows, slices)
    # volume is currently (rows, cols, slices); NIfTI wants (x, y, z)
    volume_nii = np.transpose(volume, (1, 0, 2))   # (cols, rows, slices)

    img = nib.Nifti1Image(volume_nii, affine)

    # Set voxel zooms from affine (voxel sizes)
    vox_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    img.header.set_zooms(vox_sizes)
    img.header.set_data_dtype(np.float32)

    nib.save(img, output_path)
    print(f"         saved  →  {output_path}  (shape {volume_nii.shape})")
    return volume_nii.shape


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_reverse_pipeline(root_folder: str, output_folder: str):

    if not os.path.isdir(root_folder):
        raise ValueError(f"ROOT_FOLDER does not exist:\n  {root_folder}")

    os.makedirs(output_folder, exist_ok=True)

    # Patient name = the root folder's basename  (e.g. "patient1")
    patient_label = os.path.basename(os.path.normpath(root_folder))

    # Discover modality sub-folders
    modality_dirs = {}
    for entry in sorted(os.listdir(root_folder)):
        full = os.path.join(root_folder, entry)
        if not os.path.isdir(full):
            continue
        # Normalise: treat t1ce and t1wce as the same modality key
        key = entry.lower()
        if key == "t1ce":
            key = "t1wce"
        if key in KNOWN_MODALITIES:
            modality_dirs[key] = full
        else:
            print(f"  ⚠  Unknown sub-folder, skipping: {entry}/")

    if not modality_dirs:
        raise RuntimeError(
            f"No recognised modality folders (t1/t1wce/t2/flair) found in:\n  {root_folder}"
        )

    print("\n" + "=" * 55)
    print(f"  Input    : {root_folder}")
    print(f"  Output   : {output_folder}")
    print(f"  Patient  : {patient_label}")
    print(f"  Modalities: {sorted(modality_dirs)}")
    print("=" * 55 + "\n")

    results = {}
    for mod, dcm_dir in sorted(modality_dirs.items()):
        out_filename = f"{patient_label}_{mod}.nii"
        out_path     = os.path.join(output_folder, out_filename)
        shape = dcm_folder_to_nii(
            dcm_folder     = dcm_dir,
            output_path    = out_path,
            modality_label = mod,
            patient_label  = patient_label,
        )
        results[mod] = out_filename
        print()

    print("=" * 55)
    print(f"  ✅  Done!  {len(results)} NIfTI files written.")
    print(f"  Output folder: {output_folder}")
    print("=" * 55)
    print()
    for mod, fname in sorted(results.items()):
        print(f"  📄  {fname}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ROOT_FOLDER or not OUTPUT_FOLDER:
        raise ValueError("Set ROOT_FOLDER and OUTPUT_FOLDER before running this script directly.")
    run_reverse_pipeline(ROOT_FOLDER, OUTPUT_FOLDER)
