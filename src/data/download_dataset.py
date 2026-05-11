"""
download_dataset.py
===================
Downloads and organises all training data for the Anti-UAV Drone Detection project.

Sources
-------
1. DUT Anti-UAV  – 10,000 RGB images, 35+ drone models, MIT license
   https://github.com/wangdongdut/DUT-Anti-UAV

2. Roboflow Universe – Pre-labelled YOLO-format drone images (requires API key)
   https://universe.roboflow.com

3. DUT tracking videos – 20 RGB sequences used for ByteTrack evaluation
   (part of the same DUT Anti-UAV repo)

Usage
-----
    # Download everything (requires ROBOFLOW_API_KEY in .env or env var)
    python -m src.data.download_dataset --all

    # DUT only (no API key needed)
    python -m src.data.download_dataset --dut-only

    # Roboflow only
    python -m src.data.download_dataset --roboflow-only

    # Verify an existing download without re-downloading
    python -m src.data.download_dataset --verify

Environment variables
---------------------
    ROBOFLOW_API_KEY   – Your Roboflow API key (get free at roboflow.com)
    DATA_ROOT          – Root folder for data (default: ./data)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

# Sub-directories that will be created
DIRS = {
    "dut_raw":        DATA_ROOT / "raw" / "dut_anti_uav",
    "roboflow_raw":   DATA_ROOT / "raw" / "roboflow",
    "tracking_raw":   DATA_ROOT / "raw" / "dut_tracking",
    "merged_images":  DATA_ROOT / "merged" / "images",
    "merged_labels":  DATA_ROOT / "merged" / "labels",
    "splits":         DATA_ROOT / "splits",
}

# DUT Anti-UAV dataset details
DUT_GITHUB_REPO = "https://github.com/wangdongdut/DUT-Anti-UAV"
DUT_DETECTION_URL = (
    "https://github.com/wangdongdut/DUT-Anti-UAV/releases/download/v1.0/"
    "DUT-Anti-UAV-Detection.zip"
)
DUT_TRACKING_URL = (
    "https://github.com/wangdongdut/DUT-Anti-UAV/releases/download/v1.0/"
    "DUT-Anti-UAV-Tracking.zip"
)

# Roboflow datasets to pull (workspace / project / version)
ROBOFLOW_DATASETS: List[Dict] = [
    {
        "name":      "drone-detection-primary",
        "workspace": "artificial-intelligence-nzz1a",
        "project":   "drone-dataset-6w7eq",
        "version":   2,
        "format":    "yolov11",
    },
    {
        "name":      "drone-detection-secondary",
        "workspace": "drone-detection-pexej",
        "project":   "drone-detection-data-set-yolov7",
        "version":   1,
        "format":    "yolov11",
    },
]

# Expected approximate file counts after download
EXPECTED_COUNTS = {
    "dut_detection": 10_000,
    "roboflow_total": 2_000,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_dirs() -> None:
    """Create all required directories."""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        log.debug("Ensured dir: %s", path)


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path) -> Path:
    """
    Download *url* to *dest* with a simple progress indicator.
    Uses curl (widely available) so we don't need the requests library.
    Falls back to urllib if curl is not on PATH.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl"):
        log.info("Downloading (curl): %s", url)
        subprocess.run(
            ["curl", "-L", "--progress-bar", "-o", str(dest), url],
            check=True,
        )
    else:
        import urllib.request
        log.info("Downloading (urllib): %s", url)
        with urllib.request.urlopen(url) as response:  # nosec B310
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {pct:5.1f}%  {downloaded>>20} MB / {total>>20} MB",
                              end="", flush=True)
            print()

    log.info("Saved to: %s  (%d MB)", dest, dest.stat().st_size >> 20)
    return dest


def _unzip(archive: Path, dest: Path) -> Path:
    """Extract a zip archive, showing progress."""
    log.info("Extracting %s -> %s", archive.name, dest)
    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.infolist()
        for i, member in enumerate(members, 1):
            zf.extract(member, dest)
            if i % 500 == 0 or i == len(members):
                print(f"\r  {i}/{len(members)} files extracted", end="", flush=True)
    print()
    return dest


def count_images(folder: Path, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)


def count_labels(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*.txt"))


# ---------------------------------------------------------------------------
# DUT Anti-UAV download
# ---------------------------------------------------------------------------

def download_dut_detection(force: bool = False) -> Path:
    """
    Download the DUT Anti-UAV detection subset.

    The official repo provides a zip that unpacks to:
        DUT-Anti-UAV-Detection/
            train/  images/  labels/
            val/    images/  labels/
            test/   images/  labels/

    Returns path to the unpacked root.

    NOTE: If the direct release URL is unavailable (the repo doesn't always
    publish GitHub releases), we provide fallback instructions to clone via git.
    """
    dest_root = DIRS["dut_raw"]
    unpacked = dest_root / "DUT-Anti-UAV-Detection"

    if unpacked.exists() and not force:
        n = count_images(unpacked)
        log.info("DUT detection already downloaded (%d images). Use --force to re-download.", n)
        return unpacked

    zip_path = dest_root / "DUT-Anti-UAV-Detection.zip"

    try:
        _download_with_progress(DUT_DETECTION_URL, zip_path)
        _unzip(zip_path, dest_root)
        zip_path.unlink()  # save disk space
    except Exception as exc:
        log.warning("Direct download failed (%s). Trying git clone fallback...", exc)
        _dut_git_clone_fallback(dest_root)

    n = count_images(unpacked)
    log.info("DUT detection ready: %d images in %s", n, unpacked)
    return unpacked


def _dut_git_clone_fallback(dest_root: Path) -> None:
    """
    Fallback: clone the DUT repo and look for dataset inside.
    If git is unavailable, print manual instructions and exit.
    """
    if not shutil.which("git"):
        _print_manual_dut_instructions()
        sys.exit(1)

    clone_dest = dest_root / "DUT-Anti-UAV-repo"
    if not clone_dest.exists():
        log.info("Cloning DUT Anti-UAV repo (this may take a while)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", DUT_GITHUB_REPO, str(clone_dest)],
            check=True,
        )
    else:
        log.info("Repo already cloned at %s", clone_dest)

    # The repo README explains the dataset is hosted externally.
    # Print instructions for the user.
    _print_manual_dut_instructions()


def _print_manual_dut_instructions() -> None:
    msg = """
╔══════════════════════════════════════════════════════════════════════╗
║  MANUAL STEP REQUIRED — DUT Anti-UAV Dataset                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Visit: https://github.com/wangdongdut/DUT-Anti-UAV               ║
║  2. Follow the download instructions in the README                   ║
║  3. Download the detection dataset zip                               ║
║  4. Place it at:                                                     ║
║       data/raw/dut_anti_uav/DUT-Anti-UAV-Detection.zip              ║
║  5. Re-run this script — it will detect and extract it               ║
║                                                                      ║
║  The dataset is ~2GB.  License: MIT                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(msg)


def download_dut_tracking(force: bool = False) -> Path:
    """
    Download the 20 DUT tracking video sequences used for ByteTrack evaluation.
    Returns path to the unpacked tracking root.
    """
    dest_root = DIRS["tracking_raw"]
    unpacked = dest_root / "DUT-Anti-UAV-Tracking"

    if unpacked.exists() and not force:
        seq_count = sum(1 for p in unpacked.iterdir() if p.is_dir())
        log.info("DUT tracking already downloaded (%d sequences).", seq_count)
        return unpacked

    zip_path = dest_root / "DUT-Anti-UAV-Tracking.zip"

    try:
        _download_with_progress(DUT_TRACKING_URL, zip_path)
        _unzip(zip_path, dest_root)
        zip_path.unlink()
    except Exception as exc:
        log.warning("Tracking download failed (%s).", exc)
        _print_manual_tracking_instructions()

    return unpacked


def _print_manual_tracking_instructions() -> None:
    msg = """
╔══════════════════════════════════════════════════════════════════════╗
║  MANUAL STEP — DUT Anti-UAV Tracking Sequences                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Download the tracking dataset from the DUT Anti-UAV GitHub repo    ║
║  and place the zip at:                                               ║
║    data/raw/dut_tracking/DUT-Anti-UAV-Tracking.zip                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(msg)


# ---------------------------------------------------------------------------
# Roboflow download
# ---------------------------------------------------------------------------

def download_roboflow(force: bool = False) -> List[Path]:
    """
    Download all configured Roboflow datasets.
    Requires ROBOFLOW_API_KEY environment variable.

    Returns list of downloaded dataset root paths.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        log.warning(
            "ROBOFLOW_API_KEY not set. Skipping Roboflow download.\n"
            "  Set it with: export ROBOFLOW_API_KEY=your_key_here\n"
            "  Get a free key at: https://roboflow.com"
        )
        return []

    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        log.error(
            "roboflow package not installed.\n"
            "  Install with: pip install roboflow"
        )
        return []

    rf = Roboflow(api_key=api_key)
    downloaded_paths = []

    for ds in ROBOFLOW_DATASETS:
        dest = DIRS["roboflow_raw"] / ds["name"]

        if dest.exists() and not force:
            n = count_images(dest)
            log.info("Roboflow '%s' already downloaded (%d images).", ds["name"], n)
            downloaded_paths.append(dest)
            continue

        log.info("Downloading Roboflow dataset: %s", ds["name"])
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            version.download(ds["format"], location=str(dest))
            n = count_images(dest)
            log.info("Downloaded %d images -> %s", n, dest)
            downloaded_paths.append(dest)
        except Exception as exc:
            log.error("Failed to download '%s': %s", ds["name"], exc)

    return downloaded_paths


# ---------------------------------------------------------------------------
# Label conversion helpers
# ---------------------------------------------------------------------------

def convert_dut_labels_to_yolo(dut_root: Path) -> None:
    import xml.etree.ElementTree as ET
    import shutil

    for split in ("train", "val", "test"):
        # 1. Handle the "Russian Doll" double folder
        split_root = dut_root / split
        if (split_root / split).exists():
            split_root = split_root / split

        img_dir = split_root / "img"
        xml_dir = split_root / "xml"

        # Where the rest of the script expects them to be
        final_img_dir = dut_root / split / "images"
        lbl_dir = dut_root / split / "labels"

        if not xml_dir.exists():
            continue

        final_img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        log.info("Converting %s XML annotations -> YOLO format...", split)

        # 2. Move images to the correct 'images' folder
        if img_dir.exists():
            for img_file in img_dir.iterdir():
                dest = final_img_dir / img_file.name
                if img_file.is_file() and not dest.exists():
                    shutil.move(str(img_file), str(dest))

        # 3. Convert Pascal VOC XML to YOLO TXT format
        converted = 0
        for xml_file in xml_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)  # nosec B314
                root = tree.getroot()
                size = root.find("size")
                W = float(size.find("width").text)
                H = float(size.find("height").text)

                lines = []
                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    # Center X, Center Y, Width, Height (normalized 0-1)
                    cx = ((xmin + xmax) / 2) / W
                    cy = ((ymin + ymax) / 2) / H
                    w = (xmax - xmin) / W
                    h = (ymax - ymin) / H

                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                txt_path = lbl_dir / f"{xml_file.stem}.txt"
                txt_path.write_text("\n".join(lines))
                converted += 1
            except Exception as e:
                log.warning("Failed to parse %s: %s", xml_file.name, e)

        log.info("  Converted %d label files for split '%s'.", converted, split)


# ---------------------------------------------------------------------------
# Dataset merging
# ---------------------------------------------------------------------------

def merge_datasets(
    dut_root: Path,
    roboflow_paths: List[Path],
    prefix_dut:       str = "dut",
    prefix_roboflow:  str = "rf",
) -> Tuple[int, int]:
    """
    Copy all images + labels from DUT (train split only — val/test kept separate)
    and Roboflow datasets into data/merged/{images,labels}.

    Returns (total_images, total_labels).

    Strategy
    --------
    - DUT val and test splits are KEPT separate (used as fixed eval sets).
    - DUT train images go into the merge pool.
    - All Roboflow images (all splits) go into the merge pool.
    - Final 70/15/15 split is done by split_data.py using the merged pool.
    - Files are renamed with prefixes to avoid collisions:
        dut_000001.jpg, rf_a_000001.jpg, etc.
    """
    img_out = DIRS["merged_images"]
    lbl_out = DIRS["merged_labels"]
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_img = 0
    total_lbl = 0

    def _copy_split(img_dir: Path, lbl_dir: Path, prefix: str) -> Tuple[int, int]:
        nonlocal total_img, total_lbl
        n_img = n_lbl = 0
        if not img_dir.exists():
            return 0, 0
        images = sorted(p for p in img_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, img_path in enumerate(images, 1):
            new_stem   = f"{prefix}_{i:06d}"
            new_img    = img_out / f"{new_stem}{img_path.suffix.lower()}"
            new_lbl    = lbl_out / f"{new_stem}.txt"
            shutil.copy2(img_path, new_img)
            n_img += 1
            # Copy corresponding label if it exists
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, new_lbl)
                n_lbl += 1
            else:
                # Write empty label (background / no drone image)
                new_lbl.write_text("")
        return n_img, n_lbl

    # --- DUT train split ---
    log.info("Merging DUT train split...")
    ni, nl = _copy_split(
        dut_root / "train" / "images",
        dut_root / "train" / "labels",
        prefix_dut,
    )
    log.info("  DUT train: %d images, %d labels", ni, nl)
    total_img += ni
    total_lbl += nl

    # --- Roboflow datasets (all splits merged into pool) ---
    for rf_path in roboflow_paths:
        rf_name = rf_path.name.replace("-", "_")[:20]
        for split_name in ("train", "valid", "test", "images"):
            img_d = rf_path / split_name / "images"
            lbl_d = rf_path / split_name / "labels"
            if not img_d.exists():
                img_d = rf_path / split_name  # some datasets flatten structure
                lbl_d = rf_path / "labels" / split_name
            if img_d.exists():
                pfx = f"{prefix_roboflow}_{rf_name}_{split_name[:3]}"
                ni, nl = _copy_split(img_d, lbl_d, pfx)
                log.info("  Roboflow %s/%s: %d images, %d labels",
                         rf_path.name, split_name, ni, nl)
                total_img += ni
                total_lbl += nl

    log.info("Merge complete: %d images, %d labels in %s",
             total_img, total_lbl, DIRS["merged"])
    return total_img, total_lbl


# ---------------------------------------------------------------------------
# Dataset YAML generation
# ---------------------------------------------------------------------------

def write_dataset_yaml(
    train_txt: Path,
    val_txt:   Path,
    test_txt:  Path,
    out_path:  Optional[Path] = None,
) -> Path:
    """
    Write the YOLO dataset.yaml used by Ultralytics training.
    Points to the three split .txt files (each line = image path).
    """
    out_path = out_path or (DATA_ROOT / "dataset.yaml")

    content = f"""# Anti-UAV Drone Detection — Dataset Configuration
# Generated by download_dataset.py
# Do NOT manually edit paths — re-run the script to regenerate.

path:  {DATA_ROOT.resolve()}
train: {train_txt.resolve()}
val:   {val_txt.resolve()}
test:  {test_txt.resolve()}

# Number of classes
nc: 1

# Class names
names:
  0: drone
"""
    out_path.write_text(content)
    log.info("Wrote dataset.yaml -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_download() -> bool:
    """
    Verify the download by checking image counts and label consistency.
    Returns True if everything looks correct.
    """
    log.info("=" * 60)
    log.info("VERIFICATION REPORT")
    log.info("=" * 60)
    ok = True

    # Check merged pool
    n_img = count_images(DIRS["merged_images"])
    n_lbl = count_labels(DIRS["merged_labels"])
    log.info("Merged pool   : %6d images  |  %6d labels", n_img, n_lbl)
    if n_img == 0:
        log.error("No merged images found — run with --all first.")
        ok = False
    if abs(n_img - n_lbl) > n_img * 0.05:
        log.warning("Label count differs from image count by >5%% — check your data.")

    # Check split files
    for split in ("train", "val", "test"):
        txt = DIRS["splits"] / f"{split}.txt"
        if txt.exists():
            lines = [l for l in txt.read_text().splitlines() if l.strip()]
            log.info("Split %-6s  : %6d images listed in %s", split, len(lines), txt)
        else:
            log.warning("Split file missing: %s — run split_data.py", txt)

    # Check tracking sequences
    tracking_root = DIRS["tracking_raw"] / "DUT-Anti-UAV-Tracking"
    if tracking_root.exists():
        seqs = [d for d in tracking_root.iterdir() if d.is_dir()]
        log.info("Tracking seqs : %6d sequences in %s", len(seqs), tracking_root)
    else:
        log.warning("Tracking sequences not found — tracking evaluation will be skipped.")

    log.info("=" * 60)
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare Anti-UAV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--all",           action="store_true", help="Download all sources")
    p.add_argument("--dut-only",      action="store_true", help="Download DUT only")
    p.add_argument("--roboflow-only", action="store_true", help="Download Roboflow only")
    p.add_argument("--tracking",      action="store_true", help="Download DUT tracking videos")
    p.add_argument("--merge",         action="store_true", help="Merge downloaded datasets")
    p.add_argument("--verify",        action="store_true", help="Verify existing download")
    p.add_argument("--force",         action="store_true", help="Re-download even if exists")
    p.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="Root directory for data storage (default: ./data)",
    )
    return p.parse_args()


def main() -> None:
    global DATA_ROOT, DIRS

    args = parse_args()
    DATA_ROOT = Path(args.data_root)

    # Rebuild DIRS with potentially updated DATA_ROOT
    DIRS.update({
        "dut_raw":        DATA_ROOT / "raw" / "dut_anti_uav",
        "roboflow_raw":   DATA_ROOT / "raw" / "roboflow",
        "tracking_raw":   DATA_ROOT / "raw" / "dut_tracking",
        "merged_images":  DATA_ROOT / "merged" / "images",
        "merged_labels":  DATA_ROOT / "merged" / "labels",
        "splits":         DATA_ROOT / "splits",
        "merged":         DATA_ROOT / "merged",
    })

    _make_dirs()

    dut_root        = None
    roboflow_paths  = []

    if args.verify:
        verify_download()
        return

    if args.all or args.dut_only:
        dut_root = download_dut_detection(force=args.force)
        convert_dut_labels_to_yolo(dut_root)

    if args.all or args.tracking:
        download_dut_tracking(force=args.force)

    if args.all or args.roboflow_only:
        roboflow_paths = download_roboflow(force=args.force)

    if args.all or args.merge:
        if dut_root is None:
            dut_root = DIRS["dut_raw"] / "DUT-Anti-UAV-Detection"
        if not dut_root.exists():
            log.error("DUT data not found at %s — run --dut-only first.", dut_root)
            sys.exit(1)
        merge_datasets(dut_root, roboflow_paths)
        log.info(
            "\nNext step: run split_data.py to create 70/15/15 splits\n"
            "  python -m src.data.split_data\n"
        )

    verify_download()


if __name__ == "__main__":
    main()