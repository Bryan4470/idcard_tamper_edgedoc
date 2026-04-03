# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# 
"""Train Tiny-UNet head on **pre-computed** TruFor maps (train/val CSV).

2025-07-04 – updated to accept *separate* CSVs and loader params you supplied.

Example:
```bash
python train.py \
  --train-csv FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-train.csv \
  --val-csv   FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-test.csv \
  --base-dir  FANTASY/FantasyIDiap-ICCV25-Challenge \
  --trufor-dir FANTASY/FantasyIDiap-ICCV25-Challenge/TRUFOROUTPUT \
  --mask-dir   FANTASY/FantasyIDiap-ICCV25-Challenge/GTMASKS \
  --epochs 25 --batch 1 --device cuda --workers 4 --pin-memory \
  --out-dir checkpoints
```
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloadertrufor import make_dataloader
from dataloader_tamper import make_tamper_dataloader, make_tamper_dataloaders_split

import logging

logger = logging.getLogger(__name__)


import timm, torch
import torch.nn as nn, torch.nn.functional as F
# from timm.layers import DepthwiseSeparableConv, CoordAtt

from timm.models.layers import SeparableConv2d

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            SeparableConv2d(in_ch, out_ch, kernel_size=3, padding=1),
            LayerNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, xl, xs):         # low-res, skip
        xl = self.up(xl)
        # pad if sizes differ (odd spatial dims)
        if xl.size()[2:] != xs.size()[2:]:
            xl = nn.functional.pad(
                xl,
                [0, xs.size(3) - xl.size(3),
                 0, xs.size(2) - xl.size(2)]
            )
        return self.conv(torch.cat([xs, xl], dim=1))
class TinyDocNetEdgeNeXt(nn.Module):
    def __init__(self, in_chans: int = 2):
        super().__init__()
        # ------------------------------------------------------------------
        # Encoder: EdgeNeXt-XX-Small (Pytorch-TIMM, ImageNet-22k pre-train)
        # chs = [24, 48, 88, 168]
        # ------------------------------------------------------------------
        self.enc = timm.create_model(
            'edgenext_xx_small',
            pretrained=True,
            features_only=True,
            in_chans=in_chans
        )
        c1, c2, c3, c4 = self.enc.feature_info.channels()

        # ------------------------------------------------------------------
        # Lightweight decoder – channel widths chosen to stay <2 M params
        # ------------------------------------------------------------------
        self.up3 = Up(c4 + c3, 128)         # (168+88)→128
        self.up2 = Up(128 + c2, 64)         # (128+48)→64
        self.up1 = Up(64  + c1, 32)         # (64 +24)→32

        # Heads
        self.mask_head = nn.Conv2d(32, 1, 1)            # segmentation
        self.cls_head  = nn.Sequential(                 # page-level score
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 1)        # 168 → 1
        )

        # self.cls_head  = nn.Sequential(                 # page-level score (experimental)
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Dropout(0.4),
        #     nn.Linear(c4, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, 1),
        # )

    def forward(self, x):
        H, W = x.shape[2:]                 # remember original size
        # x: (B, 3, H, W) — [green(0), npp(1), conf(2)]
        f1, f2, f3, f4 = self.enc(x)

        x = self.up3(f4, f3)               # 1/32 → 1/16
        x = self.up2(x,  f2)               # 1/16 → 1/8
        x = self.up1(x,  f1)               # 1/8  → 1/4

        mask  = torch.sigmoid(self.mask_head(x))
        mask  = F.interpolate(mask, size=(H, W), mode="nearest")  # ★ NEW ★

        score = torch.sigmoid(self.cls_head(f4)).squeeze(1)
        return mask, score



# ---------------------------------------------------------------------------
#  Loss
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, tgt: torch.Tensor, eps=1e-6):
    pred = pred.flatten(1)
    tgt = tgt.flatten(1)
    inter = (pred * tgt).sum(1)
    union = pred.sum(1) + tgt.sum(1) + eps
    return 1 - ((2 * inter + eps) / union).mean()


class CombinedLoss(nn.Module):
    def __init__(self, mask_weight=3.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mask_weight = mask_weight

    def forward(self, m_pred, m_gt, s_pred, cls_gt):
        if m_gt.shape != m_pred.shape:
            m_gt = F.interpolate(m_gt, size=m_pred.shape[-2:], mode='nearest')
        l_mask = self.bce(m_pred, m_gt) + dice_loss(m_pred, m_gt)
        l_cls = self.bce(s_pred, cls_gt)
        return self.mask_weight * l_mask + l_cls, {"mask": l_mask, "cls": l_cls}

# ---------------------------------------------------------------------------
#  Epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, opt, device, train: bool):
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    total_mask = 0.0
    total_cls = 0.0
    all_labels = []
    all_scores = []
    phase = "train" if train else "val"
    pbar = tqdm(loader, desc=f"[{phase}]", leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(train):
        for i, (maps, mask, label) in enumerate(pbar, 1):
            maps, mask, label = maps.to(device), mask.to(device), label.to(device)
            if train:
                opt.zero_grad()
            m_pred, s_pred = model(maps)
            loss, dict_loss = criterion(m_pred, mask, s_pred, label)
            if train:
                loss.backward()
                opt.step()
            total += loss.item() * maps.size(0)
            total_mask += dict_loss['mask'].item() * maps.size(0)
            total_cls += dict_loss['cls'].item() * maps.size(0)
            all_labels.extend(label.cpu().numpy().tolist())
            all_scores.extend(s_pred.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{total/i:.4f}", mask=f"{total_mask/i:.4f}", cls=f"{total_cls/i:.4f}")

    avg_loss = total / len(loader.dataset)
    avg_mask = total_mask / len(loader.dataset)
    avg_cls  = total_cls  / len(loader.dataset)

    preds = [1 if s >= 0.5 else 0 for s in all_scores]
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else float('nan')

    # FAR: attacks accepted as bonafide / total attacks
    # FRR: bonafide rejected as attacks / total bonafide
    attacks  = [l for l in all_labels if l == 1]
    bonafide = [l for l in all_labels if l == 0]
    far = sum(1 for l, p in zip(all_labels, preds) if l == 1 and p == 0) / max(len(attacks), 1)
    frr = sum(1 for l, p in zip(all_labels, preds) if l == 0 and p == 1) / max(len(bonafide), 1)

    msg = (f"[{phase}] loss={avg_loss:.4f}  mask={avg_mask:.4f}  cls={avg_cls:.4f}"
           f"  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  far={far:.4f}  frr={frr:.4f}")
    print(msg)
    logger.info(msg)
    return avg_loss, auc, f1

# ---------------------------------------------------------------------------
#  Utils
# ---------------------------------------------------------------------------

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args):
    set_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.tamper_data:
        # Split train folder into train/val — keeps GT masks for both splits
        train_loader, val_loader = make_tamper_dataloaders_split(
            data_root=args.tamper_data / "train",
            trufor_dir=args.trufor_dir / "train",
            val_split=args.val_split,
            batch_size=args.batch,
            val_batch_size=args.val_batch,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            seed=args.seed,
        )
    else:
        train_loader = make_dataloader(
            csv_file=args.train_csv,
            base_dir=args.base_dir,
            trufor_dir=args.trufor_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
        )
        val_loader = make_dataloader(
            csv_file=args.val_csv,
            base_dir=args.base_dir,
            trufor_dir=args.trufor_dir,
            mask_dir=args.mask_dir,
            batch_size=args.val_batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    model = TinyDocNetEdgeNeXt(in_chans=3).to(device)
    # print("Input channels: blue(2), npp(3), conf(4)  →  x[:, 2:5, :, :]")
    criterion = CombinedLoss(mask_weight=args.mask_weight)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    best_auc = 0.0
    best_f1  = 0.0
    start_ep = 1

    resume_ckpt = args.out_dir / "last.pth"
    if resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        best     = ckpt["best"]
        best_auc = ckpt.get("best_auc", 0.0)
        best_f1  = ckpt.get("best_f1",  0.0)
        start_ep = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']} (best loss={best:.4f} auc={best_auc:.4f} f1={best_f1:.4f})")
        print(f"Resumed from epoch {ckpt['epoch']} (best loss={best:.4f} auc={best_auc:.4f} f1={best_f1:.4f})")

    for ep in range(start_ep, args.epochs + 1):
        tr, tr_auc, tr_f1 = _run_epoch(model, train_loader, criterion, opt, device, train=True)
        val, val_auc, val_f1 = _run_epoch(model, val_loader, criterion, opt, device, train=False)
        sched.step()
        print(f"Epoch {ep:02d}/{args.epochs} – train {tr:.4f}  val {val:.4f}  val_auc={val_auc:.4f}  val_f1={val_f1:.4f}")

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best": best,
            "best_auc": best_auc,
            "best_f1": best_f1,
        }, args.out_dir / "last.pth")

        if val < best:
            best = val
            torch.save(model.state_dict(), args.out_dir / "best.pth")
            print("  ↳ new best (val loss) saved")

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), args.out_dir / "best_auc.pth")
            print(f"  ↳ new best AUC={val_auc:.4f} saved")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.out_dir / "best_f1.pth")
            print(f"  ↳ new best F1={val_f1:.4f} saved")

    print(f"Done. Best val loss={best:.4f}  AUC={best_auc:.4f}  F1={best_f1:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # --- tamper_data_with_mask mode (no CSV required) ---
    ap.add_argument("--tamper-data", type=Path, default=None,
                    help="Root of tamper_data_with_mask dataset (e.g. "
                         "/home/bryancfk/tamper_data_with_mask). "
                         "When set, --train-csv / --val-csv / --base-dir / "
                         "--mask-dir are ignored.")
    # --- original CSV mode ---
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--val-csv", type=Path, default=None)
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--mask-dir", type=Path, default=None)
    ap.add_argument("--trufor-dir", required=True, type=Path)

    ap.add_argument("--val-split", type=float, default=0.2,
                    help="Fraction of train data to use as validation (default 0.2)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--val-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base-ch", type=int, default=2)
    ap.add_argument("--mask-weight", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))

    main(ap.parse_args())