# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

from collections import namedtuple
from pathlib import Path
import numpy as np
import torch as pt
import torch.nn as nn
from trufor.cmx.builder_np_conf import (
    myEncoderDecoder as TruForModel,
)
from typing import Union


EXTRA = namedtuple(
    "EXTRA",
    [
        "BACKBONE",
        "DECODER",
        "DECODER_EMBED_DIM",
        "PREPRC",
        "BN_EPS",
        "BN_MOMENTUM",
        "DETECTION",
        "CONF",
    ],
)

MODEL = namedtuple("MODEL", ["NAME", "MODS", "EXTRA", "PRETRAINED"])
DATASET = namedtuple("DATASET", ["NUM_CLASSES"])
CONFIG = namedtuple("CONFIG", ["MODEL", "DATASET"])

DEFAULT_CONFIG = CONFIG(
    DATASET=DATASET(NUM_CLASSES=2),
    MODEL=MODEL(
        NAME="detconfcmx",
        MODS=["RGB", "NP++"],
        PRETRAINED="",
        EXTRA=EXTRA(
            BACKBONE="mit_b2",
            DECODER="MLPDecoder",
            DECODER_EMBED_DIM=512,
            PREPRC="imagenet",
            BN_EPS=0.001,
            BN_MOMENTUM=0.1,
            DETECTION="confpool",
            CONF=True,
        ),
    ),
)


def preprocess_image(img: np.ndarray) -> pt.Tensor:
    """TruFor specific preprocessing of the image."""
    img = img.astype(np.float32) / 256
    # Convert to NCHW format
    img = np.moveaxis(img, 2, 0)
    return pt.from_numpy(img)

def deprocess_image(tensor: Union[pt.Tensor, np.ndarray]) -> np.ndarray:
    """
    Exact inverse of `preprocess_image`.

    Input  : float32 C×H×W Tensor/*array*
    Output : uint8 H×W×3 numpy array (RGB)
    """
    if isinstance(tensor, pt.Tensor):
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("Expected C×H×W tensor with C==3")
        data = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        data = tensor
    else:
        raise TypeError("tensor must be torch.Tensor or numpy.ndarray")

    img = np.moveaxis(data, 0, 2) * 256.0         # CHW → HWC, back-scale
    img = np.clip(img, 0, 255).astype(np.uint8)   # guard against fp noise
    return img



import timm, torch
import torch.nn as nn, torch.nn.functional as F
# from timm.layers import DepthwiseSeparableConv, CoordAtt

from timm.models.layers import SeparableConv2d
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
            pretrained=False,
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


class TruFor:
    """Trufor model interface"""

    def __init__(
        self,
        model_path: str = Path(__file__).parent.parent / "weights/trufor.pth.tar",
        device: str = "",
    ):
        self.model_path = model_path
        self.device = device or "cuda" if pt.cuda.is_available() else "cpu"
        print(f"Model inference will run on {self.device}")
        self._model = None


        self.tinydoc = TinyDocNetEdgeNeXt(in_chans=2)


        self.tinydocpath= Path(__file__).parent.parent / "weights/tinydocedgepretrained_nobn_2channel.pth"


        state_dict = torch.load(self.tinydocpath, map_location="cpu")


        self.tinydoc.load_state_dict(state_dict, strict=True)
        self.tinydoc.eval()
        self.tinydoc.to(self.device)


    @property
    def model(self) -> nn.Module:
        """Load the model."""
        if self._model is None:
            checkpoint = pt.load(
                self.model_path, map_location=self.device, weights_only=False
            )
            model = TruForModel(cfg=DEFAULT_CONFIG).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            self._model = model.eval()

        return self._model

    def _forward(self, batch: pt.Tensor) -> tuple[pt.Tensor, ...]:
        """Run forward: -> mask_pred, conf, det, npp"""
        with pt.inference_mode():
            batch = batch.to(device=self.device)
            device_data = pt.as_tensor(batch, device=self.device)
            return self.model(device_data)

    def detect(self, img: pt.Tensor) -> float:
        """Run prediction."""
        batch = img[None, ...]
        _, _, det, _ = self._forward(batch=batch)
        return self._compute_score(det)

    def _compute_score(self, det):
        score = pt.sigmoid(det).numpy(force=True)[0]
        # Model outputs 0 for pristine pixels and 1 for forged pixels
        return float(1.0 - score)

    def localize(self, img: pt.Tensor) -> np.ndarray:
        """Run prediction."""
        batch = img[None, ...]
        # pred: [bs, 2, H, W]
        pred, _, _, _ = self._forward(batch=batch)
        return self._compute_mask(pred)

    def _compute_mask(self, pred):
        mask = pt.softmax(pred, dim=1)
        # Pick element 0 on axis 1, probability of being bonafide
        mask = mask[0, 0].numpy(force=True)
        # return a boolean mask of the image
        return mask

    def detect_and_localize(self, img: pt.Tensor) -> tuple[float, np.ndarray]:
        """Run detection and localization in one forward pass."""
        batch = img[None, ...]
        pred, conf, det, npp = self._forward(batch=batch)
        score = self._compute_score(det)
        mask = self._compute_mask(pred)
        conf=conf.squeeze().numpy(force=True)
        npp = npp.squeeze().numpy(force=True)


        np_img=deprocess_image(img)


        concat=np.stack([mask, conf, npp[1,:,:], np_img[:,:,1]/256 ]) 
        concat=concat.astype(np.float32)

        tf_tensor = pt.from_numpy(concat)

        mask_tdoc, score_tdoc =self.tinydoc(tf_tensor.unsqueeze(0).to(self.device))

        mask_tdoc=mask_tdoc.detach().cpu().numpy().squeeze()
        score_tdoc=score_tdoc.detach().cpu().numpy().squeeze()

        mask_final = mask_tdoc >= 0.5
        score_final = 1.0-score_tdoc.item()


        # import ipdb; ipdb.set_trace()
        return score_final, mask_final