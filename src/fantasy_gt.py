# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
visualise_annotations.py

Usage
-----
# Simply view the annotations:
python visualise_annotations.py path/to/photo.jpg path/to/annotations.json

# …or save an annotated copy:
python visualise_annotations.py photo.jpg annotations.json -o annotated.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

"""Utility for generating ground‑truth masks from VIA/VGG JSON rectangle annotations.

Given an image file *image_path* and its corresponding annotations file *json_path* (in VIA
format, containing rectangle regions), *create_groundtruth_mask* builds a single‑channel
binary mask with the **same height × width** as the source image:

* Pixel value **1** → background (outside any rectangle)
* Pixel value **0** → foreground (inside an annotated rectangle)

If *save_path* is supplied the mask is saved as an 8‑bit PNG with values 0 or 255 so that
it is viewable in standard image software.

Example
-------
>>> from pathlib import Path
>>> mask = create_groundtruth_mask(
...     image_path=Path("passport.jpg"),
...     json_path=Path("passport.json"),
...     save_path=Path("passport_mask.png"))
>>> mask.shape  # (H, W)
(800, 1200)
"""

from pathlib import Path
import json
import cv2
import numpy as np


def _clip(val: int, lower: int, upper: int) -> int:
    """Clamp *val* into the inclusive range [lower, upper]."""
    return max(lower, min(val, upper))


def create_groundtruth_mask(
    image_path: Path,
    json_path: Path,
    save_path: Path | None = None,
) -> np.ndarray:
    """Create a binary mask where annotated rectangles are 0 and background is 1.

    Parameters
    ----------
    image_path : Path
        Path to the RGB image file.
    json_path : Path
        Path to the VIA/VGG‑style JSON file containing rectangle annotations.
    save_path : Path | None, default=None
        Optional path to save the mask as a PNG. When provided, the mask is saved
        with values {0, 255} for compatibility with most image viewers.

    Returns
    -------
    numpy.ndarray
        A (H, W) uint8 array with values in {0, 1}.
    """

    # 1) Load image purely to obtain its spatial dimensions.
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]

    # 2) Initialise mask to 1 everywhere (background).
    mask = np.ones((height, width), dtype=np.uint8)

    # 3) Parse JSON annotations.

    if os.path.exists(json_path):
        print(f"Loading annotations from {json_path}")
        with open(json_path, encoding="utf-8") as fp:
            data = json.load(fp)

        for region in data.get("regions", []):
            sa = region.get("shape_attributes", {})
            ra = region.get("region_attributes", {})

            # Only rectangles are handled.
            if sa.get("name") != "rect":
                continue

            # Only mark altered regions; skip if absent or not "altered".
            if ra.get("region_provenance") != "altered":
                continue

            x = int(round(sa.get("x", 0)))
            y = int(round(sa.get("y", 0)))
            w = int(round(sa.get("width", 0)))
            h = int(round(sa.get("height", 0)))

            # Compute rectangle bounds, clamped to image edges.
            x0 = _clip(x, 0, width)
            y0 = _clip(y, 0, height)
            x1 = _clip(x + w, 0, width)
            y1 = _clip(y + h, 0, height)

            # 4) Set rectangle area to 0 (foreground).
            mask[y0:y1, x0:x1] = 0


    else:

        if 'bonafide' in json_path:
            print("Assuming bonafide image, no rectangles to draw.")
        else:
            raise FileNotFoundError(
                f"Annotations file not found: {json_path}"
            )


        


    # 5) Optionally save mask for visual inspection (0/255 for visibility).
    save_mask_as_jpg(mask, save_path) if save_path else None

    return mask



from pathlib import Path
from PIL import Image
import numpy as np

def save_mask_as_jpg(mask: np.ndarray,
                     outfile: str | Path = "mask.jpg",
                     quality: int = 95) -> None:
    """
    Save a boolean (or 0/1) mask as an 8-bit grayscale JPEG.

    Parameters
    ----------
    mask     : np.ndarray[bool]
        Mask with True/False or 1/0 values.
    outfile  : str | Path
        File name for the JPEG.
    quality  : int, 1–100
        JPEG quality; higher = less compression artefacts.
    """
    # 1. Make sure it’s C-contiguous & uint8 0/255
    mask_uint8 = (np.ascontiguousarray(mask).astype(np.uint8) * 255)

    # 2. Wrap in a PIL Image, single channel (“L” = 8-bit gray)
    img = Image.fromarray(mask_uint8, mode="L")

    # 3. Save as JPEG; disable chroma subsampling to avoid down-sampling
    img.save(
        outfile,
        format="JPEG",
        quality=quality,         # 95 is usually visually lossless
        subsampling=0,           # 4:4:4 – no chroma subsampling
        optimize=True            # a little smaller on disk
    )

# __all__ = [
#     "create_groundtruth_mask",
# ]
