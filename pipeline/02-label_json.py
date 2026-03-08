import os
import json
from pathlib import Path

# ====== CONFIGURATION ======
BASE_DIR = r"C:\Users\...\dataset\labels"
IMAGES_DIR = BASE_DIR
LABELME_DIR = os.path.join(BASE_DIR, "labelme_annotations")
OUTPUT_COCO_PATH = os.path.join(BASE_DIR, "_annotations_detectron2.coco.json")
# ===========================

print("Loading LabelMe JSON files...")

labelme_files = sorted(Path(LABELME_DIR).glob("*.json"))

if not labelme_files:
    raise FileNotFoundError(f"No LabelMe JSON files found in: {LABELME_DIR}")

print(f"Found {len(labelme_files)} LabelMe files")

images = []
annotations = []
categories = []
category_name_to_id = {}
annotation_id = 1
image_id = 1

for json_path in labelme_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = data.get("imagePath")
    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")

    if not file_name:
        print(f"Skipping {json_path.name}: missing imagePath")
        continue

    image_path = os.path.join(IMAGES_DIR, file_name)
    if not os.path.exists(image_path):
        print(f"Skipping {json_path.name}: image not found -> {file_name}")
        continue

    if image_width is None or image_height is None:
        print(f"Skipping {json_path.name}: missing imageWidth or imageHeight")
        continue

    images.append({
        "id": image_id,
        "file_name": file_name,
        "width": int(image_width),
        "height": int(image_height)
    })

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue

        label = shape.get("label", "unknown")
        points = shape.get("points", [])

        if len(points) < 3:
            continue

        clean_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                continue
            x = float(point[0])
            y = float(point[1])
            clean_points.append([x, y])

        if len(clean_points) < 3:
            continue

        if label not in category_name_to_id:
            category_id = len(category_name_to_id) + 1
            category_name_to_id[label] = category_id
            categories.append({
                "id": category_id,
                "name": label,
                "supercategory": "none"
            })

        category_id = category_name_to_id[label]

        segmentation = []
        for x, y in clean_points:
            segmentation.extend([x, y])

        xs = [p[0] for p in clean_points]
        ys = [p[1] for p in clean_points]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        width = x_max - x_min
        height = y_max - y_min

        area = 0.0
        n = len(clean_points)
        for i in range(n):
            x1, y1 = clean_points[i]
            x2, y2 = clean_points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2.0

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "area": area,
            "bbox": [x_min, y_min, width, height],
            "iscrowd": 0
        })

        annotation_id += 1

    image_id += 1

coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(OUTPUT_COCO_PATH, "w", encoding="utf-8") as f:
    json.dump(coco_output, f, indent=2, ensure_ascii=False)

print("Conversion completed successfully")
print("Images:", len(images))
print("Annotations:", len(annotations))
print("Categories:", len(categories))
print("COCO JSON saved at:", OUTPUT_COCO_PATH)