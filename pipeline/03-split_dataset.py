import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = Path("dataset")  # root dataset directory
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

OUTPUT_DIR = Path("dataset_split")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42
# =========================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"

random.seed(RANDOM_SEED)

images = list(IMAGES_DIR.glob("*.*"))
images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

random.shuffle(images)

total = len(images)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

print(f"Total images: {total}")
print(f"Train: {len(train_files)}")
print(f"Validation: {len(val_files)}")
print(f"Test: {len(test_files)}")

def copy_files(file_list, split_name):
    for img_path in file_list:
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        out_img_dir = OUTPUT_DIR / "images" / split_name
        out_lbl_dir = OUTPUT_DIR / "labels" / split_name

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, out_img_dir / img_path.name)

        if label_path.exists():
            shutil.copy(label_path, out_lbl_dir / label_path.name)

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Dataset successfully split.")