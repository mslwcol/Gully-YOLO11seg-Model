import os
import json
from pathlib import Path

# ====== CONFIGURATION ======
BASE_DIR = r"C:\Users\...\dataset\labels"
COCO_JSON_PATH = os.path.join(BASE_DIR, "_annotations.coco.json")
IMAGES_DIR = BASE_DIR
OUTPUT_LABELS_DIR = os.path.join(BASE_DIR, "labels")
# ===========================

print("Loading COCO JSON...")

with open(COCO_JSON_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("COCO loaded successfully")
print("Images:", len(coco.get("images", [])))
print("Annotations:", len(coco.get("annotations", [])))
print("Categories:", len(coco.get("categories", [])))

os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

category_id_to_index = {}
sorted_categories = sorted(coco["categories"], key=lambda x: x["id"])
for new_index, category in enumerate(sorted_categories):
    category_id_to_index[category["id"]] = new_index

annotations_by_image = {}
for ann in coco["annotations"]:
    image_id = ann["image_id"]
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)

for image in coco["images"]:
    image_id = image["id"]
    file_name = os.path.basename(image["file_name"])
    image_width = image["width"]
    image_height = image["height"]

    if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
        print(f"Skipping {file_name}: invalid image size")
        continue

    label_file_name = Path(file_name).stem + ".txt"
    label_file_path = os.path.join(OUTPUT_LABELS_DIR, label_file_name)

    lines = []

    for ann in annotations_by_image.get(image_id, []):
        category_id = ann["category_id"]
        class_id = category_id_to_index[category_id]

        segmentation_list = ann.get("segmentation", [])
        if not segmentation_list:
            continue

        for seg in segmentation_list:
            if not isinstance(seg, list) or len(seg) < 6:
                continue

            if len(seg) % 2 != 0:
                continue

            normalized_points = []
            valid_polygon = True

            for i in range(0, len(seg), 2):
                x = float(seg[i]) / float(image_width)
                y = float(seg[i + 1]) / float(image_height)

                if x < 0:
                    x = 0.0
                elif x > 1:
                    x = 1.0

                if y < 0:
                    y = 0.0
                elif y > 1:
                    y = 1.0

                normalized_points.extend([x, y])

            if len(normalized_points) < 6:
                valid_polygon = False

            if not valid_polygon:
                continue

            line = str(class_id) + " " + " ".join(f"{value:.6f}" for value in normalized_points)
            lines.append(line)

    with open(label_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("Conversion completed successfully")
print("YOLO segmentation labels saved in:", OUTPUT_LABELS_DIR)