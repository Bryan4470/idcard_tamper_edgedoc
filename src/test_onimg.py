# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Run full-pipeline inference (TruFor + TinyDocNetEdgeNeXt) on raw images.

No pre-computed TruFor features needed — processes images directly.
Saves per-image predicted masks and scores to --out-dir.

Usage
-----
python test_onimg.py \
    --test-dir /home/bryancfk/tamper_data_with_mask/test \
    --out-dir  test_output \
    --device   cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from edgedoc import preprocess_image, TruFor

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)


def _build_samples(test_dir: Path) -> list[dict]:
    samples = []
    for d in [test_dir / "tamper", test_dir / "tamper" / "image"]:
        if d.is_dir():
            for p in _find_images(d):
                samples.append({"path": p, "label": 0})  # 0=attack convention
            break
    for d in [test_dir / "genuine", test_dir / "genuine" / "image"]:
        if d.is_dir():
            for p in _find_images(d):
                samples.append({"path": p, "label": 1})  # 1=bonafide convention
            break
    # flat folder — no labels
    if not samples:
        for p in _find_images(test_dir):
            samples.append({"path": p, "label": None})
    return samples


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TruFor(device=args.device)

    samples = _build_samples(Path(args.test_dir))
    if not samples:
        raise RuntimeError(f"No images found under {args.test_dir}")

    has_labels = samples[0]["label"] is not None
    print(f"Found {len(samples)} images" + (
        f" ({sum(s['label']==0 for s in samples)} tamper, "
        f"{sum(s['label']==1 for s in samples)} genuine)"
        if has_labels else " (no labels — inference only)"
    ))

    all_scores, all_preds, all_gt = [], [], []
    score_lines = ["image_path\tscore\tpredicted\tactual\n"]

    for s in tqdm(samples, desc="Inference"):
        p, gt = s["path"], s["label"]
        try:
            np_img = np.array(Image.open(p).convert("RGB"))
            img_tensor = preprocess_image(np_img)
            score, mask = model.detect_and_localize(img_tensor)
            pred = int(score >= 0.5)

            all_scores.append(float(score))
            all_preds.append(pred)
            if gt is not None:
                all_gt.append(gt)

            tag = ("BON" if gt == 1 else "ATK") if gt is not None else "???"
            print(f"[{tag}] {p.name}  score={score:.4f}  pred={'bonafide' if pred==1 else 'attack'}")

            Image.fromarray(mask.astype(np.uint8) * 255).save(out_dir / f"mask_{p.stem}.png")
            actual_str = ("bonafide" if gt == 1 else "attack") if gt is not None else "unknown"
            pred_str   = "bonafide" if pred == 1 else "attack"
            score_lines.append(f"{p}\t{score:.6f}\t{pred_str}\t{actual_str}\n")

        except Exception as e:
            print(f"[WARN] Skipped {p.name}: {e}")

    scores_txt = out_dir / "scores.txt"
    with open(scores_txt, "w") as f:
        f.writelines(score_lines)
    print(f"\nScores saved to {scores_txt}")

    if not has_labels:
        print(f"Masks saved to {out_dir}/")
        return

    gt_arr    = np.array(all_gt)
    pred_arr  = np.array(all_preds)
    score_arr = np.array(all_scores)

    tn, fp, fn, tp = confusion_matrix(gt_arr, pred_arr).ravel()
    far = fp / max(tn + fp, 1)
    frr = fn / max(fn + tp, 1)
    auc = roc_auc_score(gt_arr, score_arr) if len(set(all_gt)) > 1 else float("nan")

    print(f"\n{'='*50}")
    print(f"  Total : {len(gt_arr)}  (tamper={int((gt_arr==0).sum())}, genuine={int((gt_arr==1).sum())})")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(gt_arr, pred_arr):.4f}")
    print(f"  Precision : {precision_score(gt_arr, pred_arr, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(gt_arr, pred_arr, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(gt_arr, pred_arr, zero_division=0):.4f}")
    print(f"  AUC       : {auc:.4f}")
    print(f"  FAR       : {far:.4f}  (genuine flagged as tamper)")
    print(f"  FRR       : {frr:.4f}  (tamper missed as genuine)")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*50}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=str, required=True,
                    help="test split root, e.g. .../tamper_data_with_mask/test")
    ap.add_argument("--out-dir",  type=str, default="test_output",
                    help="where to save predicted masks and scores")
    ap.add_argument("--device",   type=str, default="cuda")
    main(ap.parse_args())
