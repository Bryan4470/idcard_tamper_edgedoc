"""Pre-compute TruFor features for the tamper_data_with_mask dataset.

Walks the two sub-folders of a split (``tamper/image/`` and ``genuine/``)
and saves one ``.npy`` file per image in the mirrored output directory.

Each ``.npy`` stores a float32 array of shape ``(3, H, W)`` containing the
three per-pixel TruFor maps:
    channel 0 – localization map  (bonafide probability, from softmax)
    channel 1 – confidence map
    channel 2 – noise-print map

Usage
-----
python extract_trufor_tamper.py \\
    --data-root  /home/bryancfk/tamper_data_with_mask/train \\
    --out-dir    /home/bryancfk/tamper_data_with_mask/TRUFOROUTPUT/train \\
    --device     cuda:0

Run once for train, once for test:
    python extract_trufor_tamper.py \\
        --data-root  /home/bryancfk/tamper_data_with_mask/test \\
        --out-dir    /home/bryancfk/tamper_data_with_mask/TRUFOROUTPUT/test \\
        --device     cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

# TruFor model lives in the same src/ package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_gttrufor import TruFor, preprocess_image   # noqa: E402

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_DIM = 1259 #1259x800 original ic card size  


def _find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)


def extract_split(data_root: Path, out_dir: Path, model: TruFor) -> None:
    """Process tamper/image/ and genuine/ under *data_root*."""

    # Collect (image_path, relative_sub_path) pairs
    pairs: list[tuple[Path, str]] = []

    # handle both tamper/image/ (train) and tamper/ flat (test, no mask)
    for tamper_dir in [data_root / "tamper" / "image", data_root / "tamper"]:
        if tamper_dir.is_dir():
            for p in _find_images(tamper_dir):
                pairs.append((p, p.name))
            break

    # handle both genuine/image/ (train) and genuine/ flat (test, no mask)
    for genuine_dir in [data_root / "genuine" / "image", data_root / "genuine"]:
        if genuine_dir.is_dir():
            for p in _find_images(genuine_dir):
                pairs.append((p, p.name))
            break

    if not pairs:
        raise RuntimeError(f"No images found under {data_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, rel_name in tqdm(pairs, desc=str(data_root.name)):
        out_path = (out_dir / rel_name).with_suffix(".npy")
        if out_path.exists():
            continue   # resume-friendly: skip already computed

        try:
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            if max(w, h) > MAX_DIM:
                scale = MAX_DIM / max(w, h)
                pil_img = pil_img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )
            np_img = np.array(pil_img)
            img_tensor = preprocess_image(np_img)

            _, mask_map, conf_map, npp_map = model.detect_and_localize(img_tensor)

            # conf/npp may be at a different resolution or have extra channel dims
            h, w = mask_map.shape
            def _resize(arr, th, tw):
                arr = np.squeeze(arr)
                if arr.ndim == 0:
                    return np.full((th, tw), float(arr), dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=0)   # (C,H,W) -> (H,W)
                if arr.shape == (th, tw):
                    return arr
                return zoom(arr, (th / arr.shape[0], tw / arr.shape[1]), order=1)

            conf_map = _resize(conf_map, h, w)
            npp_map  = _resize(npp_map,  h, w)

            # stack into (3, H, W) float32
            stacked = np.stack(
                [
                    mask_map.astype(np.float32),
                    conf_map.astype(np.float32),
                    npp_map.astype(np.float32),
                ],
                axis=0,
            )
            np.save(out_path, stacked)

        except Exception as exc:
            print(f"[WARN] Failed on {img_path}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract TruFor features for tamper_data_with_mask"
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Split root, e.g. .../tamper_data_with_mask/train",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to save .npy files, e.g. .../TRUFOROUTPUT/train",
    )
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    args = ap.parse_args()

    model = TruFor(device=args.device)
    extract_split(args.data_root, args.out_dir, model)
    print("Done.")


if __name__ == "__main__":
    main()
