# line_dot_marker.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


class LineDotMarker:
    """
    Detect connected line components in a binary mask and overlay colored dots.

    …  (docstring unchanged for brevity) …
    """

    # ───────────────────────── initialization ──────────────────────────
    def __init__(
        self,
        dot_radius: int = 5,
        alpha: float = 0.8,
        color_map: str = "hsv",
        dot_density: float = 0.01,
        min_dots: int = 1,
        random_seed: int = 42,
    ):
        self.dot_radius = dot_radius
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.color_map = color_map
        self.dot_density = max(0.0, dot_density)
        self.min_dots = max(1, min_dots)
        self.rng = np.random.default_rng(random_seed)

    # ───────────────────────── private helpers ─────────────────────────
    def _generate_colors(self, n: int) -> list[tuple[int, int, int]]:
        if self.color_map == "random":
            return [
                tuple(int(c) for c in self.rng.integers(0, 256, size=3)) for _ in range(n)
            ]
        # HSV sweep
        colors: List[tuple[int, int, int]] = []
        for i in range(n):
            hue = int(180 * i / max(1, n - 1))
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors

    # ───────────────────────── core analytics ──────────────────────────
    def get_line_dot_data(self, mask: np.ndarray) -> list[dict]:
        bin_mask = (mask > 0).astype(np.uint8)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )
        colors = self._generate_colors(num - 1)
        data: List[dict] = []
        for i in range(1, num):
            coords = np.column_stack(np.where(labels == i))
            length = len(coords)
            if length == 0:
                continue
            num_dots = max(self.min_dots, int(length * self.dot_density))
            idx = self.rng.choice(length, size=min(num_dots, length), replace=False)
            pts = [(int(coords[k][1]), int(coords[k][0])) for k in idx]
            data.append({"line": i, "color": colors[i - 1], "positions": pts})
        return data

    def mark_lines(
        self,
        mask: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return BGR image with colored dots drawn on `background` (or black)."""
        data = self.get_line_dot_data(mask)
        h, w = mask.shape
        output = background.copy() if background is not None else np.zeros((h, w, 3), np.uint8)
        for item in data:
            for pt in item["positions"]:
                cv2.circle(output, pt, self.dot_radius, item["color"], -1)
        if background is not None and self.alpha < 1.0:
            output = cv2.addWeighted(output, self.alpha, background, 1 - self.alpha, 0)
        return output

    # ───────────────────────── CSV utility ─────────────────────────────
    @staticmethod
    def _write_csv(data: list[dict], csv_path: Path) -> None:
        max_dots = max(len(item["positions"]) for item in data)
        header = ["Line_Number", "Color"] + [f"Dot_{i+1}_Position" for i in range(max_dots)]
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for item in data:
                row = [item["line"], str(item["color"])]
                row += [f"({x},{y})" for x, y in item["positions"]]
                row += [""] * (max_dots - len(item["positions"]))
                writer.writerow(row)

    # ───────────────────────── high-level API  ─────────────────────────
    def process_path(
        self,
        path: str | Path,
        out_dir: str | Path = "out",
        *,
        background_paths: Iterable[str | Path] | None = None,
        save_vis: bool = False,
        img_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif"),
    ) -> list[dict]:
        """
        Run the marker on **one file or an entire folder**.

        Args
        ----
        path            : Mask file path **or** directory of mask files.
        out_dir         : Where CSV (and optional visualisations) go.
        background_paths: Optional iterable of backgrounds (same order as masks).
        save_vis        : If True, also write an image with dots (PNG).
        img_suffixes    : File endings regarded as images in folder mode.

        Returns
        -------
        results – list of dict: one entry per processed image
                 {'mask': Path, 'csv': Path, 'vis': Path|None, 'data': list[dict]}
        """
        path = Path(path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            mask_files = [path]
        else:
            mask_files = [p for p in path.rglob("*") if p.suffix.lower() in img_suffixes]

        if background_paths:
            bg_list = list(background_paths)
            if len(bg_list) != len(mask_files):
                raise ValueError("Number of backgrounds must match number of masks.")
        else:
            bg_list = [None] * len(mask_files)

        results = []
        for mask_file, bg_file in zip(mask_files, bg_list):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Could not read mask: {mask_file}")

            background = None
            if bg_file:
                background = cv2.imread(str(bg_file), cv2.IMREAD_COLOR)
                if background is None:
                    raise IOError(f"Could not read background: {bg_file}")

            data = self.get_line_dot_data(mask)

            # Save CSV
            csv_path = out_dir / f"{mask_file.stem}.csv"
            self._write_csv(data, csv_path)

            # Save visual overlay (if requested)
            vis_path = None
            if save_vis:
                vis_image = self.mark_lines(mask, background)
                vis_path = out_dir / f"{mask_file.stem}_dots.png"
                cv2.imwrite(str(vis_path), vis_image)

            results.append(
                {
                    "mask": mask_file,
                    "csv": csv_path,
                    "vis": vis_path,
                    "data": data,
                }
            )

        return results



