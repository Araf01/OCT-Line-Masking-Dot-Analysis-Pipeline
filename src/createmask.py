import os
import cv2
import numpy as np
from skimage.filters import meijering

class OCTMaskGenerator:
    """
    Generate binary masks from *already‑loaded* grayscale images.

    Parameters
    ----------
    output_dir : str | os.PathLike
        Where the masks will be written as <name>_mask.png.
    min_area : int, optional
        Small connected components (< min_area px) are discarded.
    """

    def __init__(self, output_dir, *, min_area: int = 25):
        self.output_dir = str(output_dir)
        self.min_area = int(min_area)
        os.makedirs(self.output_dir, exist_ok=True)

    # ────────────────────────────── core pipeline ──────────────────────────
    @staticmethod
    def _pipeline(img: np.ndarray) -> np.ndarray:
        """
        img : uint8 2‑D ndarray (grayscale)
        Returns a binary mask (uint8, {0,255}).
        """
        # denoise → Meijering → Otsu → closing
        denoised = cv2.fastNlMeansDenoising(img, None, h=7,
                                            templateWindowSize=7,
                                            searchWindowSize=21)
        enhanced = meijering(denoised.astype(np.float32) / 255.0, sigmas=[1])
        _, binary = cv2.threshold(
            (enhanced * 255).astype(np.uint8),
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    def _fill_components(self, binary: np.ndarray) -> np.ndarray:
        """remove tiny blobs & fill remaining components"""
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        mask = np.zeros_like(binary, dtype=np.uint8)
        for lbl_id in range(1, num):
            if stats[lbl_id, cv2.CC_STAT_AREA] >= self.min_area:
                mask[lbl == lbl_id] = 255
        return mask

    # ───────────────────────────── public API ──────────────────────────────
    def create_masks(self, items):
        """
        items : iterable[tuple[str, np.ndarray]]
            The list that ImageEnhancer.process() returns.
        Saves masks to self.output_dir and returns a list of (name, mask).
        """
        results = []
        for name, img in items:
            if img is None or img.ndim != 2:
                print(f"⚠️ Skipping {name}: not a 2‑D grayscale frame")
                continue
            binary  = self._pipeline(img)
            filled  = self._fill_components(binary)
            outpath = os.path.join(self.output_dir, f"{name}_mask.png")
            cv2.imwrite(outpath, filled)
            print("  saved", os.path.basename(outpath))
            results.append((name, filled))
        print(" All masks have been successfully saved to", self.output_dir)
        return results
