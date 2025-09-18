# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: MIT
"""
Light-weight self-contained checks for the TruFor model implementation.
Run with:  python test_trufor_manual.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import math
import numpy as np
from PIL import Image

from edgedoc import preprocess_image, TruFor    # noqa: E402


# from model_default import preprocess_image, TruFor    # noqa: E402


from fantasy_viewer import draw_annotations




# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# TEST_IMAGES_DIR = Path("/home/anjith2006/Learning/baseline-docker/test_images")
# # Feel free to point these to whatever images you like.
# IMAGE_PATHS = [
#     TEST_IMAGES_DIR / "pristine1.jpg",
#     TEST_IMAGES_DIR / "pristine2.jpg",
#     TEST_IMAGES_DIR / "tampered1.png",
#     TEST_IMAGES_DIR / "tampered2.png",
# ]

IMAGE_PATHS=['FANTASY/FantasyIDiap-ICCV25-Challenge/bonafide/huawei/french-114_03.jpg','FANTASY//FantasyIDiap-ICCV25-Challenge/attack/digital_2/huawei/arabic-01_0002_000-c7338db1_1.jpg']
DET_TOLERANCE = 1e-3       # allowed diff between detect & detect_and_localize


# --------------------------------------------------------------------------- #
# Helper routines                                                             #
# --------------------------------------------------------------------------- #

def _say_ok(msg: str) -> None:
    print(f"✓ {msg}")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"✗ {msg}")


# --------------------------------------------------------------------------- #
# 1.  preprocess_image                                                        #
# --------------------------------------------------------------------------- #

# Synthetic 10×10 RGB image filled with mid-gray
# _dummy_np = (np.ones((10, 10, 3), dtype=np.uint8) * 128)
# _t = preprocess_image(_dummy_np)

# _assert(_t.shape == (3, 10, 10), "preprocess_image – wrong tensor shape")
# _assert(_t.dtype == np.float32, "preprocess_image – wrong dtype")
# _assert(math.isclose(float(_t.max()), 0.5, rel_tol=1e-5),
#         "preprocess_image – wrong value range (expected around 0.5)")
# _say_ok("preprocess_image")


# --------------------------------------------------------------------------- #
# 2.  TruFor inference                                                        #
# --------------------------------------------------------------------------- #

model = TruFor()   # will automatically pick CPU if CUDA unavailable

labels=['bonafide', 'attack']
for p, l in zip(IMAGE_PATHS, labels):
    # if not p.exists():
    #     sys.exit(f"✗ Test image not found: {p}")

    # --------------------------------------------------------------------- #
    # 2.1  Load & preprocess                                                #
    # --------------------------------------------------------------------- #
    np_img = np.array(Image.open(p).convert("RGB"))
    img_tensor = preprocess_image(np_img)


    score_dl, mask_dl = model.detect_and_localize(img_tensor)


    print("score_dl:", score_dl)
    print("mask_dl:", mask_dl.shape)

    # import ipdb; ipdb.set_trace()

    # plot the binary mask
    # import matplotlib.pyplot as plt
    # plt.imshow(mask_dl, cmap='gray')
    # plt.title(f"Mask for ")
    # plt.axis('off')
    # plt.show()
    
    #save mask_dl with labelname
    
    mask_path = Path(f"mask_{l}.png")
    Image.fromarray(mask_dl.astype(np.uint8) * 255).save(mask_path)
    print(f"Mask saved to {mask_path}")
    save_path = Path(f"mask_annotated{l}.png")
    
    
    try:
    
        draw_annotations(mask_path, p.replace('.jpg', '.json'), save_path)
        
    except Exception as e:
        print(f"Error drawing annotations: {e}")
    

