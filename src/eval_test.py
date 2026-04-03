"""Evaluate trained TinyDocNetEdgeNeXt on the test split.

Uses pre-computed TruFor .npy features (no GT masks required).

Usage
-----
python eval_test.py \
    --test-dir   /home/bryancfk/tamper_data_with_mask/test \
    --trufor-dir /home/bryancfk/tamper_data_with_mask/TRUFOROUTPUT/test \
    --checkpoint /home/bryancfk/code.edgedcoc_iccv2025_2/checkpoints/best.pth \
    --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

import csv
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import TinyDocNetEdgeNeXt   # reuse model definition

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)


def _build_samples(test_dir: Path) -> list[dict]:
    """Walk test/tamper/ (label=1) and test/genuine/ (label=0)."""
    samples = []

    # tamper — try image/ subfolder first (train layout), then flat (test layout)
    for d in [test_dir / "tamper" / "image", test_dir / "tamper"]:
        if d.is_dir():
            imgs = _find_images(d)
            if imgs:
                for p in imgs:
                    samples.append({"img_path": p, "label": 1})
                break

    # genuine — try image/ subfolder first (train layout), then flat (test layout)
    for d in [test_dir / "genuine" / "image", test_dir / "genuine"]:
        if d.is_dir():
            imgs = _find_images(d)
            if imgs:
                for p in imgs:
                    samples.append({"img_path": p, "label": 0})
                break

    return samples


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = TinyDocNetEdgeNeXt(in_chans=3).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    samples = _build_samples(args.test_dir)
    if not samples:
        raise RuntimeError(f"No images found under {args.test_dir}")
    print(f"Found {len(samples)} test images "
          f"({sum(s['label']==1 for s in samples)} tamper, "
          f"{sum(s['label']==0 for s in samples)} genuine)")

    trufor_dir = Path(args.trufor_dir)
    labels, scores, img_paths = [], [], []

    with torch.no_grad():
        for s in tqdm(samples, desc="Inference"):
            npy_path = (trufor_dir / s["img_path"].name).with_suffix(".npy")
            try:
                tf_arr = np.load(npy_path, allow_pickle=False).astype(np.float32)
                # tf_arr: [loc(unused), conf, npp]  shape (3, H, W)
                conf_map = tf_arr[1]
                npp_map  = tf_arr[2]
                H, W = tf_arr.shape[1], tf_arr.shape[2]
                rgb_np = np.array(
                    Image.open(s["img_path"]).convert("RGB").resize((W, H), Image.LANCZOS),
                    dtype=np.float32,
                ) / 256.0  # (H, W, 3)
                combined = np.stack(
                    [rgb_np[:, :, 1], npp_map, conf_map],   # [green, npp, conf]
                    axis=0,
                ).astype(np.float32)
                tf_tensor = torch.from_numpy(combined).unsqueeze(0).to(device)  # (1,3,H,W)
                _, score = model(tf_tensor)
                labels.append(s["label"])
                scores.append(float(score.cpu()))
                img_paths.append(str(s["img_path"]))
            except Exception as e:
                print(f"[WARN] Skipped {s['img_path'].name}: {e}")

    labels = np.array(labels)
    scores = np.array(scores)
    preds  = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    far = fn / max(fn + tp, 1)   # tamper accepted as genuine / total tamper
    frr = fp / max(tn + fp, 1)   # genuine rejected as tamper / total genuine

    print(f"\n{'='*50}")
    print(f"  Total : {len(labels)}  (tamper={int((labels==1).sum())}, genuine={int((labels==0).sum())})")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(labels, preds):.4f}")
    print(f"  Precision : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(labels, preds, zero_division=0):.4f}")
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    print(f"  AUC       : {auc:.4f}")
    print(f"  FAR       : {far:.4f}  (tamper accepted as genuine)")
    print(f"  FRR       : {frr:.4f}  (genuine rejected as tamper)")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'='*50}")

    tamper_scores  = scores[labels == 1]
    genuine_scores = scores[labels == 0]
    print(f"\nScore distribution (1=tampered, 0=genuine):")
    print(f"  Tamper  — min={tamper_scores.min():.4f}  max={tamper_scores.max():.4f}  mean={tamper_scores.mean():.4f}")
    print(f"  Genuine — min={genuine_scores.min():.4f}  max={genuine_scores.max():.4f}  mean={genuine_scores.mean():.4f}")

    # Save per-image results to CSV
    csv_path = Path(args.checkpoint).parent / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "actual", "predicted", "score", "correct"])
        for path, gt, pred, score in zip(img_paths, labels.tolist(), preds.tolist(), scores.tolist()):
            actual_str    = "tamper"  if gt   == 1 else "genuine"
            predicted_str = "tamper"  if pred == 1 else "genuine"
            writer.writerow([path, actual_str, predicted_str, f"{score:.4f}", gt == pred])
    print(f"\nPer-image results saved to {csv_path}")

    # Save summary metrics to CSV
    metrics_path = Path(args.checkpoint).parent / "eval_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total",     len(labels)])
        writer.writerow(["n_tamper",  int((labels == 1).sum())])
        writer.writerow(["n_genuine", int((labels == 0).sum())])
        writer.writerow(["accuracy",  f"{accuracy_score(labels, preds):.4f}"])
        writer.writerow(["precision", f"{precision_score(labels, preds, zero_division=0):.4f}"])
        writer.writerow(["recall",    f"{recall_score(labels, preds, zero_division=0):.4f}"])
        writer.writerow(["f1",        f"{f1_score(labels, preds, zero_division=0):.4f}"])
        writer.writerow(["auc",       f"{auc:.4f}"])
        writer.writerow(["far",       f"{far:.4f}"])
        writer.writerow(["frr",       f"{frr:.4f}"])
        writer.writerow(["tp", tp])
        writer.writerow(["tn", tn])
        writer.writerow(["fp", fp])
        writer.writerow(["fn", fn])
        writer.writerow(["tamper_score_min",  f"{tamper_scores.min():.4f}"])
        writer.writerow(["tamper_score_max",  f"{tamper_scores.max():.4f}"])
        writer.writerow(["tamper_score_mean", f"{tamper_scores.mean():.4f}"])
        writer.writerow(["genuine_score_min",  f"{genuine_scores.min():.4f}"])
        writer.writerow(["genuine_score_max",  f"{genuine_scores.max():.4f}"])
        writer.writerow(["genuine_score_mean", f"{genuine_scores.mean():.4f}"])
    print(f"Summary metrics saved to {metrics_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir",   type=Path, required=True,
                    help="test split root, e.g. .../tamper_data_with_mask/test")
    ap.add_argument("--trufor-dir", type=Path, required=True,
                    help="pre-computed TruFor .npy files for test split")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).parent.parent / "checkpoints/last.pth")
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
