# COCO Dataset (Mask R-CNN)

Download the dataset in COCO JSON format from Roboflow:

https://app.roboflow.com/ds/zU8B2gmjtj?key=iy1YAdroY2

After downloading and extracting the dataset, place the files in the following directory structure:

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

To train the Mask R-CNN segmentation model run:

```
python pipeline/06-model_maskrcnn.py
```

---

