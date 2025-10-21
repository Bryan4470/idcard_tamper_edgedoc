# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
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

from model_gttrufor import preprocess_image, TruFor    # noqa: E402
from fantasy_gt import create_groundtruth_mask, save_mask_as_jpg

import os


import pandas as pd


df=pd.read_csv('FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-oneset.csv')


BASEPATH = 'FANTASY/FantasyIDiap-ICCV25-Challenge'

SAVEPATH= 'FANTASY/FantasyIDiap-ICCV25-Challenge/GTMASKS'
TFORSAVEPATH= 'FANTASY/FantasyIDiap-ICCV25-Challenge/TRUFOROUTPUT'

# dullpath=os.path.join(BASEPATH, df['path']')

df['fullpath'] = df.apply(lambda row: os.path.join(BASEPATH, row['path']), axis=1)


# sample df based on the image_type column so that there would be 10 of each type
# df = df.groupby('image_type').apply(lambda x: x.sample(n=10, random_state=42)).reset_index(drop=True)
# print the df


# iterate the df to get path image_type is_attack



DET_TOLERANCE = 1e-3       # allowed diff between detect & detect_and_localize



import glob


files=glob.glob('FANTASY/FantasyIDiap-ICCV25-Challenge/*', recursive=True)

# --------------------------------------------------------------------------- #
# Helper routines                                                             #
# --------------------------------------------------------------------------- #

def _say_ok(msg: str) -> None:
    print(f"✓ {msg}")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"✗ {msg}")

model = TruFor()   # will automatically pick CPU if CUDA unavailable

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

for index, row in df.iterrows():
    
    fullpath = row['fullpath']
    map_label = lambda x: 'bonafide' if x == False else 'attack'
    
    label = map_label(row['is_attack'])
    image_type = row['image_type']
    
    
    print(f"Processing {fullpath} with label {label} and image type {image_type}")
    # if not p.exists():
    #     sys.exit(f"✗ Test image not found: {p}")

    # --------------------------------------------------------------------- #
    # 2.1  Load & preprocess                                                #
    # --------------------------------------------------------------------- #


    np_img = np.array(Image.open(fullpath).convert("RGB"))
    img_tensor = preprocess_image(np_img)


    score_dl, mask_dl, conf_dl, npp_dl = model.detect_and_localize(img_tensor)

    # import ipdb; ipdb.set_trace()


    concat=np.stack([mask_dl, conf_dl, npp_dl[1,:,:], np_img[:,:,1]/256 ])

    save_path = os.path.join(SAVEPATH, os.path.relpath(fullpath, BASEPATH))

    savepathtrufor = os.path.join(TFORSAVEPATH, os.path.relpath(fullpath, BASEPATH).replace('.jpg', '.npy'))

    os.makedirs(os.path.dirname(savepathtrufor), exist_ok=True)
    np.save(savepathtrufor, concat)


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    create_groundtruth_mask(fullpath, fullpath.replace('.jpg', '.json'), save_path)
        

    


