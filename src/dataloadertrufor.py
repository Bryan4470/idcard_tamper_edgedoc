import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PrecomputedTruForDataset(Dataset):
    """Dataset that loads pre–computed TruFor outputs + GT masks.

    Returns `(trufor_tensor, mask_tensor, label)` where
    * `trufor_tensor`: `torch.float32`, shape **3×H×W** (mask, reliability, green‑channel)
    * `mask_tensor`   : `torch.float32`, shape **1×H×W**, values in `[0, 1]`
    * `label`         : `torch.float32`, scalar (0 = bonafide, 1 = attack)

    Parameters
    ----------
    csv_file : str | Path
        Metadata CSV (must contain columns `path`, `is_attack`, `image_type`).
    base_dir : str | Path
        Root directory of the original images – only used to replicate folder
        hierarchy when constructing paths; may be omitted in *this* dataset.
    trufor_dir : str | Path
        Directory where the `.npy` files were stored.
    mask_dir : str | Path
        Directory where the ground‑truth masks (`.jpg`) were stored.
    transform : Callable, optional
        Function applied **jointly** to `(trufor_tensor, mask_tensor)` after
        loading.  It must accept and return a tuple `(maps, mask)` so that both
        are augmented consistently.
    sample_per_type : int | None
        If given, stratified random subsample of this many images per
        `image_type`.
    """

    def __init__(
        self,
        csv_file: os.PathLike | str,
        base_dir: os.PathLike | str,
        trufor_dir: os.PathLike | str,
        mask_dir: os.PathLike | str,
        transform: Optional[Callable] = None,
        sample_per_type: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_dir = Path(base_dir)
        self.trufor_dir = Path(trufor_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # -----------------------
        # 1.  Read & subsample CSV
        # -----------------------
        df = pd.read_csv(csv_file)
        if sample_per_type is not None:
            max_count = df["is_attack"].value_counts().max()

            # 2️⃣  Oversample (with replacement when needed) so every class = max_count
            df = (
                df.groupby("is_attack", group_keys=False)
                .apply(lambda g: g.sample(n=max_count,           # upscale ∧ downscale uniformly
                                            replace=len(g) < max_count,  # minority classes ⇒ replace=True
                                            random_state=42))
                .reset_index(drop=True)
            )

            # 3️⃣  Shuffle the whole set for good measure
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Optional sanity-check:
            print(df["is_attack"].value_counts())

        # Pre‑compute absolute paths
        def _tf_path(rel: str) -> Path:
            return (self.trufor_dir / rel).with_suffix(".npy")

        def _mask_path(rel: str) -> Path:
            return self.mask_dir / rel  # already .jpg

        df["trufor_path"] = df["path"].apply(_tf_path)
        df["mask_path"] = df["path"].apply(_mask_path)
        self.df = df

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- 1.  TruFor maps (3×H×W)
        tf_arr = np.load(row["trufor_path"], allow_pickle=False).astype(np.float32)
        tf_tensor = torch.from_numpy(tf_arr)

        # ---- 2.  Mask (1×H×W)
        mask_img = Image.open(row["mask_path"]).convert("L")
        mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # add channel dim

        # ---- Optional joint transform
        if self.transform is not None:
            tf_tensor, mask_tensor = self.transform(tf_tensor, mask_tensor)

        # ---- 3.  Label (scalar)
        label = torch.tensor(float(row["is_attack"]), dtype=torch.float32)

        return tf_tensor, mask_tensor, label


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    csv_file: os.PathLike | str,
    base_dir: os.PathLike | str,
    trufor_dir: os.PathLike | str,
    mask_dir: os.PathLike | str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """Create a `DataLoader` for `PrecomputedTruForDataset`."""
    ds = PrecomputedTruForDataset(
        csv_file,
        base_dir,
        trufor_dir,
        mask_dir,
        **dataset_kwargs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# # ---------------------------------------------------------------------------
# # Minimal smoke test
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Test PrecomputedTruForDataset")
#     parser.add_argument("csv", type=Path, )
#     parser.add_argument("--base-dir", required=True, type=Path)
#     parser.add_argument("--trufor-dir", required=True, type=Path)
#     parser.add_argument("--mask-dir", required=True, type=Path)
#     parser.add_argument("--batch", type=int, default=2)
#     args = parser.parse_args()

# loader = make_dataloader(
#     csv_file=args.csv,
#     base_dir=args.base_dir,
#     trufor_dir=args.trufor_dir,
#     mask_dir=args.mask_dir,
#     batch_size=args.batch,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=False,
# )
# loader = make_dataloader(
#     csv_file="FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-train.csv",
#     base_dir='FANTASY/FantasyIDiap-ICCV25-Challenge',
#     trufor_dir='FANTASY/FantasyIDiap-ICCV25-Challenge/TRUFOROUTPUT',
#     mask_dir='FANTASY/FantasyIDiap-ICCV25-Challenge/GTMASKS',
#     batch_size=1,
#     shuffle=False,
#     num_workers=0,
#     pin_memory=False,
# )
# for maps, mask, label in loader:
#     print("✅ maps", maps.shape, maps.dtype, maps.min().item(), maps.max().item())
#     print("✅ mask", mask.shape, mask.unique())
#     print("✅ label", label)
#     break
