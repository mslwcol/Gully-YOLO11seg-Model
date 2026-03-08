# Gully Instance Segmentation with YOLO11 and Mask R-CNN

This repository contains the code used to train and evaluate **instance segmentation models for gully detection in UAV imagery**.

Two deep learning models are implemented and compared:

- YOLO11 instance segmentation
- Mask R-CNN instance segmentation

The repository includes scripts for dataset preparation, augmentation, model training, and evaluation.

---

# Models

## YOLO11 Instance Segmentation

Single-stage instance segmentation architecture implemented using the Ultralytics YOLO framework.

Three variants were evaluated:

| Model | Description |
|------|-------------|
| YOLO11s | small model |
| YOLO11m | medium model |
| YOLO11l | large model |

Training configuration:

| Parameter | Value |
|----------|------|
| Epochs | 300 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Scheduler | cosine decay |
| Data augmentation | disabled |

Training script:

```bash
python pipeline/05-model_yolo11.py
```

Outputs are saved in:

```
runs_yolo11_seg/
```

---

## Mask R-CNN Instance Segmentation

Two-stage instance segmentation model implemented using **Mask R-CNN with a ResNet50-FPN backbone**.

Training configuration:

| Parameter | Value |
|----------|------|
| Backbone | ResNet50-FPN |
| Batch size | 16 |
| Iterations | 31800 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Input resolution | 640 |

Training script:

```bash
python pipeline/06-model_maskrcnn.py
```

Outputs are stored in:

```
maskrcnn_runs/
```

---

# Dataset

The dataset contains **2420 annotated UAV image tiles**.

Total annotated instances:

3379 gully objects.

Dataset split:

| Split | Images |
|------|------|
| Train | 1694 |
| Validation | 484 |
| Test | 242 |

Annotations were created using polygon segmentation.

Two annotation formats are used:

| Format | Used for |
|------|------|
| YOLO segmentation | YOLO11 |
| COCO JSON | Mask R-CNN |

---

# Pipeline

The `pipeline` directory contains scripts used to prepare the dataset and train the segmentation models.

1. Convert annotations to YOLO format  
2. Convert annotations to COCO format  
3. Split dataset  
4. Perform data augmentation  
5. Train YOLO11 segmentation model  
6. Train Mask R-CNN segmentation model

---

# Repository Structure

```
repository
│
├── pipeline
│   ├── 01-label_txt.py
│   ├── 02-label_json.py
│   ├── 03-split_dataset.py
│   ├── 04-augmentation_dataset.py
│   ├── 05-model_yolo11.py
│   └── 06-model_maskrcnn.py
│
├── dataset
│   │
│   ├── data_yolo11
│   │   │
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   │
│   │   ├── valid
│   │   │   ├── images
│   │   │   └── labels
│   │   │
│   │   ├── test
│   │   │   ├── images
│   │   │   └── labels
│   │   │
│   │   └── data.yaml
│   │
│   └── data_maskrcnn
│       │
│       ├── train
│       │   └── _annotations.coco.json
│       │
│       ├── valid
│       │   └── _annotations.coco.json
│       │
│       └── test
│           └── _annotations.coco.json
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# Requirements

Python >= 3.10

Required libraries:

```
torch
ultralytics
detectron2
opencv-python
numpy
albumentations
```

Example installation:

```bash
pip install ultralytics
pip install opencv-python
pip install albumentations
```

Detectron2 installation instructions:

https://detectron2.readthedocs.io

---

# Reproducibility

All experiments were conducted using fixed training parameters and consistent dataset partitions to ensure reproducibility.

Random seeds are fixed where possible during training.

Evaluation results are exported automatically as JSON files.

---

# License

This project is released under the MIT License.
