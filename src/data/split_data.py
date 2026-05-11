"""
split_data.py
=============
Creates a reproducible, stratified 70 / 15 / 15 train / val / test split
from the merged image pool produced by download_dataset.py.

Key design decisions
--------------------
- Stratification: images are bucketed by whether they contain a drone
  annotation (positive) or not (negative / background). Each bucket is
  split independently so the ratio is preserved across all three sets.
- Reproducibility: controlled by a fixed random seed (default 42).
- Output: three plain-text files (train.txt / val.txt / test.txt) where
  each line is the absolute path to one image.  These files feed directly
  into the Ultralytics dataset.yaml.
- DUT fixed splits: DUT's own val and test splits are optionally appended
  to the project val/test sets instead of being re-randomised, preserving
  comparability with published DUT baselines.

Usage
-----
    python -m src.data.split_data
    python -m src.data.split_data --seed 123 --train 0.70 --val 0.15
    python -m src.data.split_data --use-dut-fixed-splits --data-root ./data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_positive(label_path: Path) -> bool:
    """Return True if the label file contains at least one annotation."""
    if not label_path.exists():
        return False
    return bool(label_path.read_text().strip())


def _gather_pairs(
    img_dir: Path,
    lbl_dir: Path,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    Collect (image_path, label_path | None) pairs from a directory.
    Images without a corresponding label are kept as negative samples.
    """
    pairs = []
    img_exts = {".jpg", ".jpeg", ".png"}
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in img_exts:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def _stratified_split(
    pairs: List[Tuple[Path, Optional[Path]]],
    train_ratio: float,
    val_ratio:   float,
    seed:        int,
) -> Tuple[List, List, List]:
    """
    Split *pairs* into train / val / test preserving positive / negative ratio.

    Returns three lists of (image_path, label_path | None).
    """
    positives = [(i, l) for i, l in pairs if l is not None and is_positive(l)]
    negatives = [(i, l) for i, l in pairs if not (l is not None and is_positive(l))]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def _do_split(items):
        n = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train = items[:n_train]
        val   = items[n_train : n_train + n_val]
        test  = items[n_train + n_val :]
        return train, val, test

    pos_train, pos_val, pos_test = _do_split(positives)
    neg_train, neg_val, neg_test = _do_split(negatives)

    train = pos_train + neg_train
    val   = pos_val   + neg_val
    test  = pos_test  + neg_test

    # Shuffle each split so positives/negatives aren't grouped
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def _write_split_txt(pairs: List, out_path: Path) -> int:
    """Write image paths to a .txt file. Returns number of lines written."""
    lines = [str(img_path.resolve()) for img_path, _ in pairs]
    out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def _copy_to_split_dirs(
    pairs: List,
    split_name: str,
    splits_root: Path,
) -> None:
    """
    Optionally copy images+labels into splits/{train,val,test}/{images,labels}
    for tools that expect this layout (e.g., some RT-DETR configs).
    """
    img_out = splits_root / split_name / "images"
    lbl_out = splits_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        if lbl_path and lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / f"{img_path.stem}.txt")
        else:
            (lbl_out / f"{img_path.stem}.txt").write_text("")  # empty = background


def _write_dataset_yaml(
    data_root:  Path,
    splits_dir: Path,
    txt_train:  Path,
    txt_val:    Path,
    txt_test:   Path,
    use_dirs:   bool = False,
) -> Path:
    yaml_path = data_root / "dataset.yaml"

    if use_dirs:
        train_ref = str((splits_dir / "train" / "images").resolve())
        val_ref   = str((splits_dir / "val"   / "images").resolve())
        test_ref  = str((splits_dir / "test"  / "images").resolve())
    else:
        train_ref = str(txt_train.resolve())
        val_ref   = str(txt_val.resolve())
        test_ref  = str(txt_test.resolve())

    content = f"""# Anti-UAV Drone Detection — YOLO Dataset Configuration
# Auto-generated by split_data.py — do not edit manually.
# Re-run split_data.py to regenerate after data changes.

path: {data_root.resolve()}

train: {train_ref}
val:   {val_ref}
test:  {test_ref}

# Classes
nc: 1
names:
  0: drone

# Dataset statistics (filled by EDA notebook)
# total_images:    ~12000
# positive_images: ~10000   # images containing at least one drone
# negative_images: ~2000    # background-only images
# train_images:    ~8400
# val_images:      ~1800
# test_images:     ~1800
"""
    yaml_path.write_text(content)
    log.info("Wrote dataset.yaml -> %s", yaml_path)
    return yaml_path


def _print_split_stats(
    train: List,
    val:   List,
    test:  List,
) -> None:
    def _stats(pairs, name):
        pos = sum(1 for _, l in pairs if l is not None and is_positive(l))
        neg = len(pairs) - pos
        log.info(
            "  %-6s  total=%5d  positives=%5d (%4.1f%%)  negatives=%5d (%4.1f%%)",
            name, len(pairs),
            pos, 100 * pos / max(len(pairs), 1),
            neg, 100 * neg / max(len(pairs), 1),
        )

    log.info("=" * 66)
    log.info("SPLIT STATISTICS")
    log.info("=" * 66)
    _stats(train, "train")
    _stats(val,   "val")
    _stats(test,  "test")
    total = len(train) + len(val) + len(test)
    log.info("  Total   %d images", total)
    log.info("=" * 66)


# ---------------------------------------------------------------------------
# DUT fixed splits (optional)
# ---------------------------------------------------------------------------

def _load_dut_fixed_split(
    dut_root: Path,
    split:    str,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    Load DUT's own pre-defined split (val or test).
    These are kept fixed to allow comparison with published DUT baselines.
    """
    img_dir = dut_root / split / "images"
    lbl_dir = dut_root / split / "labels"
    if not img_dir.exists():
        log.warning("DUT %s split not found at %s", split, img_dir)
        return []
    pairs = _gather_pairs(img_dir, lbl_dir)
    log.info("Loaded %d images from DUT fixed '%s' split.", len(pairs), split)
    return pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_split(
    data_root:             Path,
    train_ratio:           float = 0.70,
    val_ratio:             float = 0.15,
    seed:                  int   = 42,
    use_dut_fixed_splits:  bool  = True,
    copy_to_split_dirs:    bool  = False,
) -> Dict[str, Path]:
    """
    Core split logic. Returns dict of {split_name: txt_path}.
    """
    merged_img = data_root / "merged" / "images"
    merged_lbl = data_root / "merged" / "labels"
    splits_dir = data_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not merged_img.exists() or not any(merged_img.iterdir()):
        raise FileNotFoundError(
            f"No merged images found at {merged_img}.\n"
            "Run download_dataset.py --all first."
        )

    log.info("Gathering image-label pairs from %s ...", merged_img)
    all_pairs = _gather_pairs(merged_img, merged_lbl)
    log.info("Found %d total images.", len(all_pairs))

    # ----------------------------------------------------------------
    # If using DUT fixed splits:
    #   - Remove DUT images from the merged pool (they were only put
    #     there if download script put DUT train into merged)
    #   - val and test come from DUT's fixed val/test
    # ----------------------------------------------------------------
    dut_root = data_root / "raw" / "dut_anti_uav" / "DUT-Anti-UAV-Detection"

    if use_dut_fixed_splits and dut_root.exists():
        log.info("Using DUT fixed val/test splits.")

        # Only the merged (non-DUT-val/test) images go into the train pool
        dut_val_images  = {
            p.name for p, _ in _load_dut_fixed_split(dut_root, "val")
        }
        dut_test_images = {
            p.name for p, _ in _load_dut_fixed_split(dut_root, "test")
        }
        excluded = dut_val_images | dut_test_images
        train_pool = [(i, l) for i, l in all_pairs if i.name not in excluded]

        rng = random.Random(seed)
        rng.shuffle(train_pool)
        train = train_pool  # all remaining go to train

        val  = _load_dut_fixed_split(dut_root, "val")
        test = _load_dut_fixed_split(dut_root, "test")

    else:
        # Standard stratified split from merged pool
        train, val, test = _stratified_split(
            all_pairs, train_ratio, val_ratio, seed
        )

    _print_split_stats(train, val, test)

    # Write .txt files
    txt_train = splits_dir / "train.txt"
    txt_val   = splits_dir / "val.txt"
    txt_test  = splits_dir / "test.txt"

    n_train = _write_split_txt(train, txt_train)
    n_val   = _write_split_txt(val,   txt_val)
    n_test  = _write_split_txt(test,  txt_test)

    log.info("Wrote split files:")
    log.info("  train.txt  %d lines -> %s", n_train, txt_train)
    log.info("  val.txt    %d lines -> %s", n_val,   txt_val)
    log.info("  test.txt   %d lines -> %s", n_test,  txt_test)

    # Write split metadata JSON for the EDA notebook
    meta = {
        "seed":        seed,
        "train_ratio": train_ratio,
        "val_ratio":   val_ratio,
        "test_ratio":  round(1 - train_ratio - val_ratio, 4),
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      n_test,
        "n_total":     n_train + n_val + n_test,
        "used_dut_fixed_splits": use_dut_fixed_splits,
        "positives": {
            "train": sum(1 for _, l in train if l and is_positive(l)),
            "val":   sum(1 for _, l in val   if l and is_positive(l)),
            "test":  sum(1 for _, l in test  if l and is_positive(l)),
        },
    }
    meta_path = splits_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Wrote metadata -> %s", meta_path)

    # Optionally copy images into split directory layout
    if copy_to_split_dirs:
        log.info("Copying images to split directories (this may take a while)...")
        _copy_to_split_dirs(train, "train", splits_dir)
        _copy_to_split_dirs(val,   "val",   splits_dir)
        _copy_to_split_dirs(test,  "test",  splits_dir)
        log.info("Done copying.")

    # Write dataset.yaml
    _write_dataset_yaml(
        data_root, splits_dir,
        txt_train, txt_val, txt_test,
        use_dirs=copy_to_split_dirs,
    )

    return {"train": txt_train, "val": txt_val, "test": txt_test}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create 70/15/15 dataset splits.")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--train", type=float, default=0.70, dest="train_ratio")
    p.add_argument("--val",   type=float, default=0.15, dest="val_ratio")
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument(
        "--use-dut-fixed-splits",
        action="store_true",
        help="Use DUT's own val/test instead of random split (recommended)",
    )
    p.add_argument(
        "--copy-to-dirs",
        action="store_true",
        help="Also copy images into splits/{train,val,test}/images layout",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_split(
        data_root=Path(args.data_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_dut_fixed_splits=args.use_dut_fixed_splits,
        copy_to_split_dirs=args.copy_to_dirs,
    )


if __name__ == "__main__":
    main()