# train_detectron.py
# Detectron2 Mask R-CNN training on GPU
# Configuration aligned to ~300 epochs for 2420 images with batch size 16
# Includes final evaluation and result saving

import os
import json

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# ===========================
# DATA PATHS
# ===========================
train_json = "C:\Users\...\dataset\data_maskrcnn\train\_annotations.coco.json"
train_imgs = "C:\Users\...\dataset\data_maskrcnn\train"

valid_json = "C:\Users\...\dataset\data_maskrcnn\valid\_annotations.coco.json"
valid_imgs = "C:\Users\...\dataset\data_maskrcnn\valid"


# ===========================
# DATASET REGISTRATION
# ===========================
def safe_register(name: str, json_path: str, img_root: str):
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    register_coco_instances(name, {}, json_path, img_root)

safe_register("gully_train", train_json, train_imgs)
safe_register("gully_val", valid_json, valid_imgs)


# ===========================
# TRAINER WITH COCO EVALUATION
# ===========================
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)


# ===========================
# MODEL CONFIGURATION
# ===========================
cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# Datasets
cfg.DATASETS.TRAIN = ("gully_train",)
cfg.DATASETS.TEST = ("gully_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# Pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

# GPU
cfg.MODEL.DEVICE = "cuda"

# ===========================
# INPUT SIZE (match YOLO)
# ===========================
cfg.INPUT.MIN_SIZE_TRAIN = (640,)
cfg.INPUT.MAX_SIZE_TRAIN = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE_TEST = 640

# ===========================
# SOLVER (aligned to YOLO)
# ===========================
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.WARMUP_ITERS = 453
cfg.SOLVER.MAX_ITER = 45300
cfg.SOLVER.STEPS = ()
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CHECKPOINT_PERIOD = 7550
cfg.TEST.EVAL_PERIOD = 7550

# ===========================
# ROI HEADS
# ===========================
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# ===========================
# OUTPUT DIRECTORY
# ===========================
cfg.OUTPUT_DIR = os.path.expanduser("~/projects/gully_runs/maskrcnn_r50_300epoch")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())

# ===========================
# TRAINING
# ===========================
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ===========================
# FINAL TEST / EVALUATION
# ===========================
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator("gully_val", cfg, distributed=False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
val_loader = build_detection_test_loader(cfg, "gully_val")
results = inference_on_dataset(trainer.model, val_loader, evaluator)

print("Final evaluation results:", results)

with open(os.path.join(cfg.OUTPUT_DIR, "final_metrics.json"), "w") as f:
    json.dump(results, f, indent=4)