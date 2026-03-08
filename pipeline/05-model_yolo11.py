from ultralytics import YOLO
import os
import json

# Disable tqdm batch progress bar
os.environ["TQDM_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

def main():
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="data.yaml",
        epochs=300,
        imgsz=640,
        rect=True,
        batch=16,
        device=0,
        workers=4,
        save=True,
        save_period=50,
        project="runs_yolo11_seg",
        name="small",

        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        cos_lr=True,

        # keep verbose True so epoch metrics print
        verbose=True,

        # disable online augmentation
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        flipud=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        degrees=0.0,
        auto_augment=None,
    )

    best_model_path = os.path.join("runs_yolo11_seg", "small", "weights", "best.pt")
    best_model = YOLO(best_model_path)

    metrics = best_model.val(
        data="data.yaml",
        imgsz=640,
        rect=True,
        batch=16,
        device=0,
        workers=4,
        split="val",
        save_json=True,
        project="runs_yolo11_seg",
        name="small_test"
    )

    results_dict = {
        "metrics/precision_B": float(metrics.box.mp),
        "metrics/recall_B": float(metrics.box.mr),
        "metrics/mAP50_B": float(metrics.box.map50),
        "metrics/mAP50-95_B": float(metrics.box.map),
        "metrics/precision_M": float(metrics.seg.mp),
        "metrics/recall_M": float(metrics.seg.mr),
        "metrics/mAP50_M": float(metrics.seg.map50),
        "metrics/mAP50-95_M": float(metrics.seg.map)
    }

    save_dir = os.path.join("runs_yolo11_seg", "small_test")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    with open(os.path.join(save_dir, "final_metrics.txt"), "w") as f:
        for k, v in results_dict.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Final evaluation results saved in:", save_dir)
    print(results_dict)

if __name__ == "__main__":
    main()