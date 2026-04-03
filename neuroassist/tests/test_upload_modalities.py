from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from routes.upload import (
    _detect_modality_token,
    _match_modality_files,
    _prepare_nifti_case_input,
)


class TestUploadModalityAssembly(unittest.TestCase):
    def test_detect_modality_token_supports_aliases(self):
        cases = {
            "BraTS20_Validation_001_t1.nii": "t1",
            "BraTS20_Validation_001_t1w.nii.gz": "t1w",
            "BraTS20_Validation_001_t1ce.nii": "t1ce",
            "BraTS20_Validation_001_t1wce.nii.gz": "t1wce",
            "BraTS20_Validation_001_t2.nii": "t2",
            "BraTS20_Validation_001_t2w.nii.gz": "t2w",
            "BraTS20_Validation_001_flair.nii": "flair",
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(_detect_modality_token(filename), expected)

    def test_match_modality_files_canonicalizes_aliases(self):
        nii_files = [
            Path("BraTS20_Validation_001_t1w.nii"),
            Path("BraTS20_Validation_001_t1wce.nii"),
            Path("BraTS20_Validation_001_t2w.nii"),
            Path("BraTS20_Validation_001_flair.nii"),
        ]

        matched = _match_modality_files(nii_files)

        self.assertEqual(set(matched), {"t1", "t1ce", "t2", "flair"})
        self.assertTrue(matched["t1"].name.endswith("_t1w.nii"))
        self.assertTrue(matched["t1ce"].name.endswith("_t1wce.nii"))
        self.assertTrue(matched["t2"].name.endswith("_t2w.nii"))
        self.assertTrue(matched["flair"].name.endswith("_flair.nii"))

    def test_prepare_nifti_case_input_builds_canonical_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nii_files = []
            for name in (
                "BraTS20_Validation_001_t1.nii",
                "BraTS20_Validation_001_t1wce.nii.gz",
                "BraTS20_Validation_001_t2w.nii",
                "BraTS20_Validation_001_flair.nii",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
                nii_files.append(path)

            case_dir = _prepare_nifti_case_input(nii_files, root)

            self.assertTrue(case_dir.is_dir())
            self.assertTrue((case_dir / "uploaded_case_t1.nii").exists())
            self.assertTrue((case_dir / "uploaded_case_t1ce.nii.gz").exists())
            self.assertTrue((case_dir / "uploaded_case_t2.nii").exists())
            self.assertTrue((case_dir / "uploaded_case_flair.nii").exists())


if __name__ == "__main__":
    unittest.main()
