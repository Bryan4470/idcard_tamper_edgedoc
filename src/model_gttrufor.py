# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Anjith George  <anjith.george@idiap.ch>
#
# SPDX-License-Identifier: MIT

from collections import namedtuple
from pathlib import Path
import numpy as np
import torch as pt
import torch.nn as nn
from trufor.cmx.builder_np_conf import (
    myEncoderDecoder as TruForModel,
)

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
        return score, mask, conf, npp