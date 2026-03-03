from ultralytics import YOLO
import torch

def train_yolo(data_yaml, epochs=15, batch_size=8, lr=0.001):
    """
    Train YOLOv8 on a dataset specified by a YAML file.
    Returns the trained YOLO model object.
    Minimal output: only a single progress bar per epoch.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n

    # Train quietly (no batch-level prints)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        lr0=lr,
        imgsz=640,
        amp=True,
        workers=2,
        device=device,
        verbose=False,  # single-line progress bar
        plots=False
    )

    return model


def evaluate_yolo(model, data_yaml, split="test"):
    """
    Evaluate YOLOv8 model on a dataset split ('train', 'val', 'test').
    Returns metrics dictionary and prints results.
    """
    results = model.val(
        data=data_yaml,
        split=split,
        workers=2,
        save_dir=f"runs/val_{split}"
    )

    metrics = {
        "mAP50": results.box.map50,
        "mAP50_95": results.box.map,
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "Images/sec": results.speed["inference"]
    }

    print(f"YOLO evaluation on {split} set:")
    for k, v in metrics.items():
        if k == "Images/sec":
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:.4f}")
    
    return metrics
