"""
NIfTI → DICOM Conversion Pipeline  (Kaggle / Jupyter compatible)
=================================================================
Converts .nii / .nii.gz files (T1, T1wCE, T2, FLAIR) into per-modality
DICOM slice folders.

Output structure:
    patient1/
    ├── t1/         slice_0001.dcm … slice_NNNN.dcm
    ├── t1wce/      slice_0001.dcm … slice_NNNN.dcm
    ├── t2/         slice_0001.dcm … slice_NNNN.dcm
    └── flair/      slice_0001.dcm … slice_NNNN.dcm

HOW TO USE (Kaggle / Jupyter):
    1. Set INPUT_FOLDER below to your folder path
    2. Set OUTPUT_PARENT to where patient1/ should be created
    3. Run the cell — that's it.

Requirements:
    pip install nibabel pydicom numpy
"""

import os
import re
import sys
import datetime
import warnings
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURE THESE TWO PATHS BEFORE RUNNING
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FOLDER  = ""   # folder with .nii files
OUTPUT_PARENT = ""                      # patient1/ will be created here

# ══════════════════════════════════════════════════════════════════════════════

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not found. Run:  !pip install nibabel")

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import generate_uid, ExplicitVRLittleEndian
except ImportError:
    sys.exit("pydicom not found. Run:  !pip install pydicom")


# ── modality detection ────────────────────────────────────────────────────────
# Order matters: t1wce/t1ce must come before plain t1
# t1ce  = BraTS / common naming convention
# t1wce = alternative naming convention
# Both map to the same output folder: t1wce
MODALITY_PATTERNS = {
    "t1wce": re.compile(r"t1(wce|ce)",               re.IGNORECASE),
    "t1":    re.compile(r"(?<![a-z])t1(?![a-z0-9])", re.IGNORECASE),
    "t2":    re.compile(r"(?<![a-z])t2(?![a-z0-9])", re.IGNORECASE),
    "flair": re.compile(r"flair",                    re.IGNORECASE),
}

def detect_modality(filename: str):
    stem = filename
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    for key, pattern in MODALITY_PATTERNS.items():
        if pattern.search(stem):
            return key
    return None


# ── normalisation ─────────────────────────────────────────────────────────────
def normalise_to_uint16(volume: np.ndarray) -> np.ndarray:
    vol = volume.astype(np.float64)
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        return np.zeros_like(vol, dtype=np.uint16)
    return ((vol - vmin) / (vmax - vmin) * 65535.0).astype(np.uint16)


# ── build one DICOM slice ─────────────────────────────────────────────────────
def build_dicom_slice(
    pixel_array,
    slice_index,
    modality_tag,
    series_uid,
    study_uid,
    study_date,
    patient_name,
    voxel_spacing,
    image_orientation,
    image_position,
):
    sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID     = generate_uid()
    file_meta.ImplementationVersionName  = "NII2DCM_PIPELINE"

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR   = False
    ds.is_little_endian = True

    ds.PatientName      = patient_name
    ds.PatientID        = patient_name
    ds.PatientBirthDate = ""
    ds.PatientSex       = ""

    ds.StudyInstanceUID       = study_uid
    ds.StudyDate              = study_date
    ds.StudyTime              = "000000"
    ds.ReferringPhysicianName = ""
    ds.StudyID                = "1"
    ds.AccessionNumber        = ""

    ds.SeriesInstanceUID  = series_uid
    ds.SeriesNumber       = 1
    ds.Modality           = "MR"
    ds.SeriesDescription  = modality_tag.upper()

    ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = slice_index + 1

    ds.Rows    = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.SamplesPerPixel           = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated             = 16
    ds.BitsStored                = 16
    ds.HighBit                   = 15
    ds.PixelRepresentation       = 0

    row_sp, col_sp, thickness = voxel_spacing
    ds.PixelSpacing             = [float(row_sp), float(col_sp)]
    ds.SliceThickness           = float(thickness)
    ds.ImageOrientationPatient  = [float(v) for v in image_orientation]
    ds.ImagePositionPatient     = [float(v) for v in image_position]
    ds.SliceLocation            = float(image_position[2])
    ds.FrameOfReferenceUID      = generate_uid()
    ds.PositionReferenceIndicator = ""

    p_min, p_max        = int(pixel_array.min()), int(pixel_array.max())
    ds.WindowCenter     = (p_min + p_max) // 2
    ds.WindowWidth      = max(p_max - p_min, 1)
    ds.RescaleIntercept = 0
    ds.RescaleSlope     = 1

    ds.PixelData = pixel_array.tobytes()
    return ds


# ── convert one NIfTI file ────────────────────────────────────────────────────
def nii_to_dicom(nii_path, output_dir, modality_tag, patient_name,
                 study_uid, study_date):
    os.makedirs(output_dir, exist_ok=True)

    img  = nib.load(nii_path)
    data = np.asarray(img.dataobj)

    if data.ndim == 4:
        warnings.warn(f"{os.path.basename(nii_path)} is 4-D — using first volume.")
        data = data[..., 0]

    data    = np.transpose(data, (2, 0, 1))   # (X,Y,Z) → (Z, rows, cols)
    vol_u16 = normalise_to_uint16(data)
    n_slices = data.shape[0]

    affine  = img.affine
    zooms   = img.header.get_zooms()
    vox_row = float(zooms[1]) if len(zooms) > 1 else 1.0
    vox_col = float(zooms[0]) if len(zooms) > 0 else 1.0
    vox_z   = float(zooms[2]) if len(zooms) > 2 else 1.0

    row_cos = affine[:3, 0] / (np.linalg.norm(affine[:3, 0]) or 1)
    col_cos = affine[:3, 1] / (np.linalg.norm(affine[:3, 1]) or 1)
    orientation = list(row_cos) + list(col_cos)

    series_uid = generate_uid()
    print(f"  [{modality_tag.upper():5s}]  {os.path.basename(nii_path)}  →  {n_slices} slices")

    for z in range(n_slices):
        pos = list((affine @ np.array([0, 0, z, 1], dtype=float))[:3])
        ds  = build_dicom_slice(
            pixel_array       = vol_u16[z],
            slice_index       = z,
            modality_tag      = modality_tag,
            series_uid        = series_uid,
            study_uid         = study_uid,
            study_date        = study_date,
            patient_name      = patient_name,
            voxel_spacing     = (vox_row, vox_col, vox_z),
            image_orientation = orientation,
            image_position    = pos,
        )
        ds.save_as(
            os.path.join(output_dir, f"slice_{z+1:04d}.dcm"),
            write_like_original=False,
        )

    return n_slices


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(input_folder: str, output_parent: str):

    if not os.path.isdir(input_folder):
        raise ValueError(f"INPUT_FOLDER does not exist:\n  {input_folder}")

    nii_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".nii", ".nii.gz"))
    ]
    if not nii_files:
        raise FileNotFoundError(f"No .nii / .nii.gz files found in:\n  {input_folder}")

    modality_map = {}
    for fname in nii_files:
        mod = detect_modality(fname)
        if mod is None:
            print(f"  ⚠  Unrecognised modality, skipping: {fname}")
            continue
        if mod in modality_map:
            print(f"  ⚠  Duplicate '{mod}', skipping: {fname}")
            continue
        modality_map[mod] = os.path.join(input_folder, fname)

    if not modality_map:
        raise RuntimeError("No recognised modalities (t1/t1wce/t2/flair) detected.")

    os.makedirs(output_parent, exist_ok=True)
    existing = [
        d for d in os.listdir(output_parent)
        if re.match(r"patient\d+$", d, re.IGNORECASE)
        and os.path.isdir(os.path.join(output_parent, d))
    ]
    patient_name = f"patient{len(existing) + 1}"
    output_root  = os.path.join(output_parent, patient_name)

    study_uid  = generate_uid()
    study_date = datetime.date.today().strftime("%Y%m%d")

    print("\n" + "=" * 55)
    print(f"  Input    : {input_folder}")
    print(f"  Output   : {output_root}")
    print(f"  Patient  : {patient_name}")
    print(f"  Modalities: {sorted(modality_map)}")
    print("=" * 55 + "\n")

    total = 0
    for mod in sorted(modality_map):
        out_dir = os.path.join(output_root, mod)
        count   = nii_to_dicom(
            nii_path     = modality_map[mod],
            output_dir   = out_dir,
            modality_tag = mod,
            patient_name = patient_name,
            study_uid    = study_uid,
            study_date   = study_date,
        )
        total += count
        print(f"        saved → {out_dir}\n")

    print("=" * 55)
    print(f"  ✅  Done!  {total} total DICOM slices written.")
    print(f"  Output root : {output_root}")
    print("=" * 55)
    print(f"\n  {patient_name}/")
    for mod in sorted(modality_map):
        print(f"  ├── {mod}/  (slice_0001.dcm … slice_NNNN.dcm)")
    print()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not INPUT_FOLDER or not OUTPUT_PARENT:
        raise ValueError("Set INPUT_FOLDER and OUTPUT_PARENT before running this script directly.")
    run_pipeline(INPUT_FOLDER, OUTPUT_PARENT)
