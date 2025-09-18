# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: MIT
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

    def forward(self, x):
        H, W = x.shape[2:]                 # remember original size
        x= x[:, 1:3, :, :]  # (B, 2, H, W) – remove alpha channel
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
        l_mask = self.bce(m_pred, m_gt) + dice_loss(m_pred, m_gt)
        l_cls = self.bce(s_pred, cls_gt)
        return self.mask_weight * l_mask + l_cls, {"mask": l_mask, "cls": l_cls}

# ---------------------------------------------------------------------------
#  Epoch helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, opt, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    with torch.set_grad_enabled(train):
        for maps, mask, label in loader:
            maps, mask, label = maps.to(device), mask.to(device), label.to(device)
            if train:
                opt.zero_grad()
            m_pred, s_pred = model(maps)
            loss, dict_loss = criterion(m_pred, mask, s_pred, label)
            if train:
                loss.backward()
                opt.step()
            total += loss.item() * maps.size(0)

            # print("dict_loss:", dict_loss)

            logger.info(f"Batch loss: {loss.item():.4f}: maploss: {dict_loss['mask'].item():.4f}, classloss: {dict_loss['cls'].item():.4f}   (train={train})")
            print(f"Batch loss: {loss.item():.4f}: maploss: {dict_loss['mask'].item():.4f}, classloss: {dict_loss['cls'].item():.4f}   (train={train})")
    print(f"Epoch {'train' if train else 'val'} loss: {total / len(loader.dataset):.4f}")
    logger.info(f"Epoch {'train' if train else 'val'} loss: {total / len(loader.dataset):.4f}")

    # Return average loss for the epoch
    return total / len(loader.dataset)

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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    model = TinyDocNetEdgeNeXt(in_chans=2).to(device)
    criterion = CombinedLoss(mask_weight=args.mask_weight)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, opt, device, train=True)
        val = _run_epoch(model, val_loader, criterion, opt, device, train=False)
        sched.step()
        print(f"Epoch {ep:02d}/{args.epochs} – train {tr:.4f}  val {val:.4f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), args.out_dir / "best.pth")
            print("  ↳ new best saved")
    print("Done. Best val loss:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, type=Path)
    ap.add_argument("--val-csv", required=True, type=Path)
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--trufor-dir", required=True, type=Path)
    ap.add_argument("--mask-dir", required=True, type=Path)

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