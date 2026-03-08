# Dataset

The dataset used for training and evaluating the segmentation models is hosted on Roboflow.

Due to GitHub storage limitations, the dataset is not included in this repository and must be downloaded separately.

Two export formats are provided to support the two model architectures implemented in this project.

---

# YOLO11 Segmentation Dataset

Download the dataset in YOLO segmentation format from Roboflow:

https://app.roboflow.com/ds/IgRIi42cl9?key=WpsBLm4zOf

After downloading and extracting the dataset, place the files in the following directory structure:

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

The `data.yaml` file contains the dataset configuration required by the YOLO11 training script.

To train the YOLO11 segmentation model run:

```
python pipeline/05-model_yolo11.py
```

---

