import os
from pathlib import Path
import cv2
import numpy as np


class MaskCleaner:
    """
    Removes spurious blobs from binary masks.

    • `clean_mask(mask)`  – original per‑image API (unchanged)
    • `clean_folder(src_dir, dst_dir=None, pattern='*.png')`
        → cleans every image in *src_dir* that matches *pattern*
          and writes <name>_clean.png into *dst_dir* (defaults next to src)
    """

    def __init__(self, min_area: int = 25):
        self.min_area = int(min_area)

    # ────────────────────────────── core logic ────────────────────────────
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return a cleaned copy of a single binary mask array."""
        binary = (mask > 0).astype(np.uint8)

        num, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)

        for lbl_id in range(1, num):  # skip background (0)
            if stats[lbl_id, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned[lbl == lbl_id] = 255
        return cleaned

    # ───────────────────────────── batch helper ───────────────────────────
    @staticmethod
    def _is_image(fname: str) -> bool:
        return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

    def clean_folder(
        self,
        src_dir: str | os.PathLike,
        dst_dir: str | os.PathLike | None = None,
        pattern: str = "*.png",
        suffix: str = "_clean",
    ):
        """
        Clean every mask file inside *src_dir* that matches *pattern*.

        Parameters
        ----------
        src_dir : str | Path
            Folder that already contains the (noisy) mask images.
        dst_dir : str | Path | None, optional
            Where to write the cleaned masks.
            • If None → a sibling folder called "<src_dir>_clean".
            • If == src_dir → clean *in‑place* by overwriting originals.
        pattern : str
            Glob pattern (e.g. '*.png' or '*.tif') to pick files.
        suffix : str
            Suffix appended before the extension (foo_mask → foo_mask_clean.png).

        Returns
        -------
        list[tuple[str, np.ndarray]]
            [(base_name_without_ext, cleaned_mask), ...]
        """
        src_path = Path(src_dir)
        if not src_path.is_dir():
            raise NotADirectoryError(src_dir)

        if dst_dir is None:
            dst_path = src_path.parent / f"{src_path.name}_clean"
        else:
            dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)

        results = []
        for file in sorted(src_path.glob(pattern)):
            if not self._is_image(file.name):
                continue
            raw = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                print(f" Could not read {file}")
                continue

            cleaned = self.clean_mask(raw)

            out_name = f"{file.stem}{suffix}{file.suffix}"
            cv2.imwrite(str(dst_path / out_name), cleaned)
            print("  saved", out_name)
            results.append((file.stem, cleaned))

        print(f" {len(results)} mask(s) cleaned and stored in {dst_path}")
        return results
