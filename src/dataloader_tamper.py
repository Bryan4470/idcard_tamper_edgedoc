"""Dataset and DataLoader for the tamper_data_with_mask folder layout.

Expected folder structure
-------------------------
data_root/
    tamper/
        image/   ← tampered images (.jpg / .png)
        mask/    ← binary masks  (user convention: white=tampered, black=genuine)
    genuine/     ← genuine images (.jpg / .png)  — NO masks needed

The masks in this dataset use the OPPOSITE convention to the rest of this
project (where white=genuine, black=tampered).  This module inverts them
automatically so everything downstream sees the project convention:

    project mask: 1.0 = genuine  |  0.0 = tampered
    user mask   : 1.0 = tampered |  0.0 = genuine  → inverted here

For genuine images a full-ones mask (all genuine) is created synthetically.

Returns
-------
(trufor_tensor, mask_tensor, label)  — identical contract to
PrecomputedTruForDataset in dataloadertrufor.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )


class TamperFolderDataset(Dataset):
    """Dataset for tamper_data_with_mask layout.

    Parameters
    ----------
    data_root : Path
        The split root, e.g. ``.../tamper_data_with_mask/train`` or
        ``.../tamper_data_with_mask/test``.
    trufor_dir : Path
        Directory that mirrors *data_root*'s structure but holds ``.npy``
        files produced by ``extract_trufor_tamper.py``.
        E.g. ``.../TRUFOROUTPUT/train``.
    transform : Callable, optional
        Joint transform applied to ``(trufor_tensor, mask_tensor)`` —
        same contract as in ``PrecomputedTruForDataset``.
    """

    def __init__(
        self,
        data_root: os.PathLike | str,
        trufor_dir: os.PathLike | str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.trufor_dir = Path(trufor_dir)
        self.transform = transform

        data_root = Path(data_root)
        tamper_img_dir = data_root / "tamper" / "image"
        tamper_msk_dir = data_root / "tamper" / "mask"
        genuine_dir = data_root / "genuine"

        self.samples: list[dict] = []

        # --- tampered samples (label = 1.0) --------------------------------
        for img_path in _find_images(tamper_img_dir):
            mask_path = tamper_msk_dir / img_path.name
            if not mask_path.exists():
                # try common cross-extension matches (.jpg ↔ .png)
                mask_path = self._find_mask_any_ext(tamper_msk_dir, img_path.stem)
            self.samples.append(
                {
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "label": 1.0,
                    "split": "tamper",
                }
            )

        # --- genuine samples (label = 0.0) ---------------------------------
        genuine_subdir = genuine_dir if genuine_dir.is_dir() else None
        # handle both flat genuine/ and genuine/image/ layouts
        if genuine_subdir is not None:
            img_subdir = genuine_subdir / "image"
            if img_subdir.is_dir():
                genuine_subdir = img_subdir
            for img_path in _find_images(genuine_subdir):
                self.samples.append(
                    {
                        "img_path": img_path,
                        "mask_path": None,   # no mask → synthesised below
                        "label": 0.0,
                        "split": "genuine",
                    }
                )

        if not self.samples:
            raise RuntimeError(
                f"No images found under {data_root}. "
                "Check that tamper/image/ and genuine/ exist."
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_mask_any_ext(folder: Path, stem: str) -> Path:
        for ext in _IMG_EXTENSIONS:
            p = folder / (stem + ext)
            if p.exists():
                return p
        # return the expected path even if missing; __getitem__ will raise
        return folder / (stem + ".jpg")

    def _trufor_path(self, img_path: Path) -> Path:
        """Mirror the image path into trufor_dir, replacing suffix with .npy."""
        return (self.trufor_dir / img_path.name).with_suffix(".npy")

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # ---- 1.  TruFor maps (3×H×W): [mask, conf, npp] -----------------
        tf_path = self._trufor_path(s["img_path"])
        tf_arr = np.load(tf_path, allow_pickle=False).astype(np.float32)
        # tf_arr[0] = trufor localization mask (unused)
        # tf_arr[1] = conf map
        # tf_arr[2] = npp map
        conf_map = tf_arr[1]   # (H, W)
        npp_map  = tf_arr[2]   # (H, W)

        H, W = tf_arr.shape[1], tf_arr.shape[2]

        # ---- 2.  Green channel from original image ----------------------
        # resize image to match TruFor output resolution
        pil_img = Image.open(s["img_path"]).convert("RGB").resize(
            (W, H), Image.LANCZOS
        )
        rgb_np    = np.array(pil_img, dtype=np.float32) / 256.0  # (H, W, 3)
        green_map = rgb_np[:, :, 1]   # (H, W)

        # ---- 3.  Stack into (3, H, W): [green, npp, conf] ---------------
        combined = np.stack(
            [green_map, npp_map, conf_map],
            axis=0,
        ).astype(np.float32)
        tf_tensor = torch.from_numpy(combined)   # (3, H, W)

        # ---- 4.  GT Mask (1×H×W) in project convention ------------------
        # project convention: 1.0 = genuine,  0.0 = tampered
        # user's   convention: 1.0 = tampered, 0.0 = genuine  (inverted!)
        if s["mask_path"] is not None:
            mask_img = Image.open(s["mask_path"]).convert("L").resize(
                (W, H), Image.NEAREST
            )
            mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0
            # invert: user white(tampered)→0.0,  user black(genuine)→1.0
            mask_np = 1.0 - mask_np
        else:
            # genuine image → all-ones mask (every pixel is genuine)
            mask_np = np.ones((H, W), dtype=np.float32)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)

        # ---- Optional joint transform ------------------------------------
        if self.transform is not None:
            tf_tensor, mask_tensor = self.transform(tf_tensor, mask_tensor)

        # ---- 5.  Label --------------------------------------------------
        label = torch.tensor(s["label"], dtype=torch.float32)

        return tf_tensor, mask_tensor, label


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_tamper_dataloader(
    data_root: os.PathLike | str,
    trufor_dir: os.PathLike | str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """Create a ``DataLoader`` backed by ``TamperFolderDataset``."""
    ds = TamperFolderDataset(data_root, trufor_dir, **dataset_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_tamper_dataloaders_split(
    data_root: os.PathLike | str,
    trufor_dir: os.PathLike | str,
    val_split: float = 0.2,
    batch_size: int = 8,
    val_batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0,
    **dataset_kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders by splitting the train folder.

    Stratified by label so the class ratio is preserved in both splits.
    Returns ``(train_loader, val_loader)``.
    """
    ds = TamperFolderDataset(data_root, trufor_dir, **dataset_kwargs)
    labels = [s["label"] for s in ds.samples]
    indices = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader
