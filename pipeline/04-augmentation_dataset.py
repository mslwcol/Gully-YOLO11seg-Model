
import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import albumentations as A


# -----------------------------
# Albumentations pipeline
# -----------------------------
def build_transform(seed: int = 42):
    # ReplayCompose ensures deterministic replay if needed
    return A.ReplayCompose(
        [
            # 1) Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                scale=(0.90, 1.10),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                shear={"x": (-5, 5), "y": (-5, 5)},
                p=0.5,
            ),

            # 2) Photometric
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.CLAHE(p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # 3) Sharpness and Noise
            A.GaussianBlur(p=0.3),
            A.ISONoise(p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# -----------------------------
# YOLO-seg label helpers
# -----------------------------
def read_yolo_seg_label(label_path: Path):
    """
    Returns:
        polys: list of dict { 'cls': int, 'pts': [(x_norm, y_norm), ...] }
    """
    polys = []
    if not label_path.exists():
        return polys

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) < 6 or len(coords) % 2 != 0:
            # Need at least 3 points (x,y)*3
            continue
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        polys.append({"cls": cls, "pts": pts})
    return polys


def write_yolo_seg_label(label_path: Path, polys, w: int, h: int):
    """
    polys: list of dict { 'cls': int, 'pts_px': [(x_px, y_px), ...] }
    Writes normalized coords.
    """
    out_lines = []
    for poly in polys:
        cls = poly["cls"]
        pts_px = poly["pts_px"]

        # Normalize
        coords = []
        for (x, y) in pts_px:
            xn = float(x) / float(w)
            yn = float(y) / float(h)
            # clamp [0,1]
            xn = min(max(xn, 0.0), 1.0)
            yn = min(max(yn, 0.0), 1.0)
            coords.extend([xn, yn])

        if len(coords) < 6:
            continue

        out_lines.append(str(cls) + " " + " ".join(f"{c:.6f}" for c in coords))

    label_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def polygon_area(pts):
    """Shoelace area for polygon pts [(x,y),...]"""
    if len(pts) < 3:
        return 0.0
    x = np.array([p[0] for p in pts], dtype=np.float32)
    y = np.array([p[1] for p in pts], dtype=np.float32)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_points(pts, w, h):
    """Clip points into image bounds."""
    clipped = []
    for (x, y) in pts:
        x = min(max(float(x), 0.0), float(w - 1))
        y = min(max(float(y), 0.0), float(h - 1))
        clipped.append((x, y))
    return clipped


# -----------------------------
# Main augmentation routine
# -----------------------------
def augment_one(image_path: Path, label_path: Path, tfm, out_img: Path, out_lbl: Path, min_area_px2=25.0):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    polys = read_yolo_seg_label(label_path)

    # Flatten polygon vertices as keypoints, keep mapping
    keypoints = []
    mapping = []  # (poly_idx, vertex_idx)
    for pi, poly in enumerate(polys):
        for vi, (xn, yn) in enumerate(poly["pts"]):
            x = xn * w
            y = yn * h
            keypoints.append((x, y))
            mapping.append((pi, vi))

    # Apply transform
    transformed = tfm(image=img, keypoints=keypoints)
    img_t = transformed["image"]
    kpts_t = transformed["keypoints"]

    # Rebuild polygons in pixel coords
    new_polys = []
    if polys:
        # initialize container
        pts_by_poly = {i: [] for i in range(len(polys))}
        for (pi, vi), (x, y) in zip(mapping, kpts_t):
            pts_by_poly[pi].append((vi, x, y))

        for pi, poly in enumerate(polys):
            # restore original vertex order
            pts_sorted = sorted(pts_by_poly[pi], key=lambda t: t[0])
            pts_px = [(x, y) for _, x, y in pts_sorted]

            # clip to bounds of transformed image
            ht, wt = img_t.shape[:2]
            pts_px = clip_points(pts_px, wt, ht)

            # basic validity
            if len(pts_px) < 3:
                continue
            if polygon_area(pts_px) < min_area_px2:
                continue

            new_polys.append({"cls": poly["cls"], "pts_px": pts_px})

        # write labels using transformed image size
        ht, wt = img_t.shape[:2]
        write_yolo_seg_label(out_lbl, new_polys, wt, ht)
    else:
        # no labels: write empty label file (still valid)
        out_lbl.write_text("", encoding="utf-8")

    # save image
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img), img_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_images", type=str, required=True, help="Input train images folder (e.g., dataset_split/images/train)")
    ap.add_argument("--in_labels", type=str, required=True, help="Input train labels folder (e.g., dataset_split/labels/train)")
    ap.add_argument("--out_images", type=str, required=True, help="Output images folder")
    ap.add_argument("--out_labels", type=str, required=True, help="Output labels folder")
    ap.add_argument("--copies", type=int, default=1, help="How many augmented copies per original")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--copy_originals", action="store_true", help="Also copy originals to output")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    in_images = Path(args.in_images)
    in_labels = Path(args.in_labels)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # collect images
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in in_images.iterdir() if p.suffix.lower() in exts]
    imgs.sort()

    tfm = build_transform(seed=args.seed)

    for img_path in imgs:
        stem = img_path.stem
        lbl_path = in_labels / f"{stem}.txt"

        # Optionally copy originals
        if args.copy_originals:
            shutil.copy(img_path, out_images / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, out_labels / lbl_path.name)
            else:
                (out_labels / f"{stem}.txt").write_text("", encoding="utf-8")

        # Create augmented copies
        for i in range(args.copies):
            out_img = out_images / f"{stem}_aug{i+1}{img_path.suffix.lower()}"
            out_lbl = out_labels / f"{stem}_aug{i+1}.txt"

            augment_one(img_path, lbl_path, tfm, out_img, out_lbl)

    print("Done. Augmented train set generated.")


if __name__ == "__main__":
    main()