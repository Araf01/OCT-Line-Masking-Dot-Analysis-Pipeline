"""
• Robust ImageEnhancer (handles float / 16-bit TIFFs)
• Returns a list of (name, image) so we never lose filenames
• PatchPredictor runs segmentation
• Masks are written as <original>_mask.png
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import psutil
import tensorflow as tf

class ImageEnhancer:
    """
    Returns: list of tuples (base_name_without_ext_or_page, enhanced_uint8_image)
    """

    def __init__(self, gamma=0.1, sigma=30, blend=0.65, blur_w=-0.5):
        self.gamma, self.sigma, self.blend, self.blur_w = gamma, sigma, blend, blur_w

    # ---------- dtype fix & enhancement helpers ---------------------------
    @staticmethod
    def _to_uint8(a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint8:
            return a.copy()
        a = np.nan_to_num(a.astype(np.float32))
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo + 1e-10:
            return np.zeros_like(a, dtype=np.uint8)
        return ((a - lo) / (hi - lo) * 255).astype(np.uint8)

    def _enhance(self, g):
        inv  = cv2.bitwise_not(g)
        eq   = cv2.equalizeHist(inv)
        gamma = (cv2.pow(eq.astype(np.float32) / 255.0, self.gamma) * 255).astype(np.uint8)
        blur  = cv2.GaussianBlur(gamma, (0, 0), self.sigma)
        return cv2.addWeighted(gamma, self.blend, blur, self.blur_w, 0)

    # ---------- loaders ----------------------------------------------------
    def _read_tiff(self, file_path):
        base = Path(file_path).stem
        with tifffile.TiffFile(file_path) as tif:
            for i, page in enumerate(tif.pages):
                img = self._to_uint8(page.asarray())
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                name = f"{base}_page{i:03d}"
                yield name, self._enhance(img)

    @staticmethod
    def _is_image(f):  # incl. tif
        return f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

    # ---------- public -----------------------------------------------------
    def process(self, path: str):
        out = []
        if os.path.isfile(path):
            if path.lower().endswith((".tif", ".tiff")):
                out.extend(self._read_tiff(path))
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    out.append((Path(path).stem, self._enhance(img)))
        elif os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if not self._is_image(fn):
                    continue
                full = os.path.join(path, fn)
                if fn.lower().endswith((".tif", ".tiff")):
                    out.extend(self._read_tiff(full))
                else:
                    img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        out.append((Path(fn).stem, self._enhance(img)))
        else:
            raise ValueError(f"{path!r} is not a file or directory")
        return out


# ──────────────────────────────────────────────────────────────────────────────
#  PatchPredictor  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class PatchPredictor:
    def __init__(self, model_path, patch_size=256, overlap=0.5):
        self.PATCH_SIZE = patch_size
        self.OVERLAP = overlap
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "focal_dice_loss": self.focal_dice_loss,
                "dice_coef": self.dice_coef,
            },
        )
        self.process = psutil.Process(os.getpid())

    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        inter = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2.0 * inter + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
        )

    def focal_dice_loss(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = 1 - self.dice_coef(y_true, y_pred)
        return bce + dice

    def _report(self, t0, m0):
        dt = time.time() - t0
        cpu = self.process.cpu_percent(None)
        ram = (self.process.memory_info().rss - m0) / (1024**2)
        print(f"⏱ {dt:.2f}s | CPU {cpu:.1f}% | +RAM {ram:.1f} MB")

    def predict_mask(self, img: np.ndarray):
        if img is None or img.ndim != 2:
            print(" Needs a 2-D grayscale image.")
            return None

        t0, m0 = time.time(), self.process.memory_info().rss
        stride = int(self.PATCH_SIZE * (1 - self.OVERLAP))
        H, W = img.shape
        imgf = img.astype(np.float32) / 255.0

        pad_h = (stride - ((H - self.PATCH_SIZE) % stride)) % stride
        pad_w = (stride - ((W - self.PATCH_SIZE) % stride)) % stride

        padded = np.pad(imgf, ((0, pad_h), (0, pad_w)))
        pred = np.zeros_like(padded, dtype=np.float32)
        cnt  = np.zeros_like(padded, dtype=np.float32)

        for y in range(0, padded.shape[0] - self.PATCH_SIZE + 1, stride):
            for x in range(0, padded.shape[1] - self.PATCH_SIZE + 1, stride):
                patch = padded[y:y+self.PATCH_SIZE, x:x+self.PATCH_SIZE]
                patch_exp = patch[None, ..., None]
                p = self.model.predict(patch_exp, verbose=0)[0, ..., 0]
                pred[y:y+self.PATCH_SIZE, x:x+self.PATCH_SIZE] += p
                cnt[y:y+self.PATCH_SIZE, x:x+self.PATCH_SIZE]  += 1

        cnt[cnt == 0] = 1e-8
        mask = (pred / cnt)[:H, :W] > 0.1
        self._report(t0, m0)
        return (mask.astype(np.uint8) * 255)



