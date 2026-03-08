# Gully Instance Segmentation with YOLO11 and Mask R-CNN

This repository contains the code used to train and evaluate **instance segmentation models for gully erosion detection in UAV imagery**.

Two deep learning models are implemented and compared:

- YOLO11 instance segmentation
- Mask R-CNN instance segmentation

The repository includes scripts for dataset preparation, augmentation, model training, and evaluation.

Due to GitHub storage limitations, the dataset is hosted externally on Roboflow.

---

# Models

## YOLO11 Instance Segmentation

Single-stage instance segmentation architecture implemented using the Ultralytics YOLO framework.

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

```
python pipeline/05-model_yolo11.py
```

---

## Mask R-CNN Instance Segmentation

Two-stage instance segmentation architecture.

Configuration:

| Parameter | Value |
|----------|------|
| Backbone | ResNet50-FPN |
| Epochs | 300 |
| Batch size | 16 |
| Learning rate | 0.01 |
| Input resolution | 640 |

Training script:

```
python pipeline/06-model_maskrcnn.py
```

---

# Dataset

The dataset used for training and evaluating the segmentation models is hosted on **Roboflow**.

Two export formats are provided to support the two model architectures implemented in this repository.

---

## YOLO11 Segmentation Dataset

Download the dataset in YOLO segmentation format:

https://app.roboflow.com/ds/IgRIi42cl9?key=WpsBLm4zOf

After downloading, place the dataset in the following directory structure:

```
dataset
└── data_yolo11
    ├── train
    │   ├── images
    │   └── labels
    │
    ├── valid
    │   ├── images
    │   └── labels
    │
    ├── test
    │   ├── images
    │   └── labels
    │
    └── data.yaml
```

This format is used to train the **YOLO11 instance segmentation model**.

---

## COCO JSON Dataset

Download the dataset in COCO instance segmentation format:

https://app.roboflow.com/ds/zU8B2gmjtj?key=iy1YAdroY2

After downloading, place the dataset in the following directory structure:

```
dataset
└── data_maskrcnn
    ├── train
    │   └── _annotations.coco.json
    │
    ├── valid
    │   └── _annotations.coco.json
    │
    └── test
        └── _annotations.coco.json
```

This format is used to train the **Mask R-CNN instance segmentation model**.

---

# Dataset Summary

Dataset characteristics:

| Property | Value |
|--------|------|
| Total images | 2420 |
| Total annotated instances | 3379 |
| Annotation type | Polygon segmentation |

Dataset split:

| Split | Images |
|------|------|
| Train | 1694 |
| Validation | 484 |
| Test | 242 |

---

# Pipeline

The `pipeline` directory contains scripts used to prepare the dataset and train the segmentation models.

1. Convert annotations to YOLO format  
2. Convert annotations to COCO format  
3. Split dataset  
4. Data augmentation  
5. Train YOLO11 model  
6. Train Mask R-CNN model  

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
│   ├── data_yolo11
│   │  └── README.md
│   └── data_maskrcnn
│      └── README.md
│
├── CITATION.cff
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

```
pip install ultralytics
pip install opencv-python
pip install albumentations
```

---

# Reproducibility

To reproduce the experiments:

1. Download the dataset from the Roboflow links above.
2. Place the dataset inside the `dataset` directory using the structure described in this README.
3. Run the training scripts located in the `pipeline` folder.

---

# Citation

If you use this repository in your research, please cite the repository using the metadata provided in the `CITATION.cff` file.

---

# License

This project is released under the MIT License.
