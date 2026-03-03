# main.py
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from torchvision import transforms
from trainRCNN import train_model
from trainYOLO import train_yolo, evaluate_yolo
from datasetsRCNN import DetectionDataset
from create_data_lists import create_pet_dataset, create_pennfudan_dataset
from config import DEFAULTS, DATASETS


# -------------------------------------------------
# Merge default config with dataset-specific config
# -------------------------------------------------
def get_config(dataset_name, model_type):
    config = DEFAULTS[model_type].copy()
    config.update(DATASETS[dataset_name][model_type])
    return config


# -------------------------------------------------
# Load RCNN dataset
# -------------------------------------------------
def load_rcnn_dataset(json_folder, split):
    json_path = os.path.join(json_folder, f"{split}.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return DetectionDataset(root=None, data_list=data)


# -------------------------------------------------
# Helper function to save qualitative images
# -------------------------------------------------
def save_qualitative_images(images, preds_boxes, preds_labels, gts_boxes, gts_labels, model_name, dataset_name, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        if i >= 5:  # only save 5 images
            break
        fig, ax = plt.subplots(1)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        ax.imshow(img)

        # Draw ground truth boxes in green
        for box, label in zip(gts_boxes[i], gts_labels[i]):
            box = box.detach().cpu().numpy() if isinstance(box, torch.Tensor) else box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)

        # Draw predicted boxes in red
        for box, label in zip(preds_boxes[i], preds_labels[i]):
            box = box.detach().cpu().numpy() if isinstance(box, torch.Tensor) else box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

        plt.axis('off')
        save_path = os.path.join(folder_path, f"{model_name}_{dataset_name}_img{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Create dataset splits
    create_pet_dataset()
    create_pennfudan_dataset()

    # Create qualitative folder
    qualitative_folder = r"C:\Users\costellodt\OneDrive - beloit.edu\Desktop\Machine Learning\Homework2\runs\qualitative"
    os.makedirs(qualitative_folder, exist_ok=True)

    summary_rows = []

    for dataset_name in DATASETS.keys():

        print(f"\n=== Training on {dataset_name} ===\n")

        # ==========================
        # RCNN
        # ==========================
        rcnn_cfg = get_config(dataset_name, "RCNN")

        train_dataset = load_rcnn_dataset(rcnn_cfg["json_folder"], "train")
        val_dataset   = load_rcnn_dataset(rcnn_cfg["json_folder"], "val")
        test_dataset  = load_rcnn_dataset(rcnn_cfg["json_folder"], "test")

        rcnn_model_metrics, rcnn_test_metrics = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_classes=rcnn_cfg["num_classes"],
            epochs=rcnn_cfg["epochs"],
            patience=rcnn_cfg["patience"],
            batch_size_train=rcnn_cfg["batch_size_train"],
            batch_size_val=rcnn_cfg["batch_size_val"],
            learning_rate=rcnn_cfg["learning_rate"],
            checkpoint_prefix=rcnn_cfg["checkpoint_prefix"]
        )

        summary_rows.append({
            "Dataset": dataset_name,
            "Model": "RCNN",
            **rcnn_test_metrics
        })

        # --------------------------
        # Save qualitative RCNN images
        # --------------------------
        # Select 5 images from test_dataset
        test_images = [test_dataset[i][0] for i in range(min(5, len(test_dataset)))]
        gts_boxes = [test_dataset[i][1]['boxes'] for i in range(min(5, len(test_dataset)))]
        gts_labels = [test_dataset[i][1]['labels'] for i in range(min(5, len(test_dataset)))]

        # Get predictions from trained RCNN model
        rcnn_model = train_model.__globals__['get_model'](rcnn_cfg["num_classes"])
        rcnn_model.load_state_dict(torch.load(f"{rcnn_cfg['checkpoint_prefix']}_best.pth"))
        rcnn_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        rcnn_model.eval()
        preds_boxes, preds_labels = [], []
        with torch.no_grad():

            transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        for img in test_images:
            img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
            img_tensor = img_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = rcnn_model([img_tensor.squeeze(0)])
            boxes = outputs[0]['boxes'].cpu()
            labels = outputs[0]['labels'].cpu()
            preds_boxes.append(boxes)
            preds_labels.append(labels)

        save_qualitative_images(
            test_images, preds_boxes, preds_labels,
            gts_boxes, gts_labels,
            model_name="RCNN", dataset_name=dataset_name,
            folder_path=qualitative_folder
        )


        # ==========================
        # YOLO
        # ==========================
        yolo_cfg = get_config(dataset_name, "YOLO")

        yolo_model = train_yolo(
            data_yaml=yolo_cfg["data_yaml"],
            epochs=yolo_cfg["epochs"],
            batch_size=yolo_cfg["batch_size"],
            lr=yolo_cfg["learning_rate"]
        )

        yolo_test_metrics = evaluate_yolo(
            yolo_model,
            yolo_cfg["data_yaml"],
            split="test"
        )

        # Rename YOLO Images/sec → Inference Speed
        if "Images/sec" in yolo_test_metrics:
            yolo_test_metrics["Inference Speed"] = yolo_test_metrics.pop("Images/sec")

        summary_rows.append({
            "Dataset": dataset_name,
            "Model": "YOLO",
            **yolo_test_metrics
        })

        # --------------------------
        # Save qualitative YOLO images
        # --------------------------
        from utils import load_yolo_test_images  # assuming helper exists to load images and ground truth
        test_images, gts_boxes, gts_labels = load_yolo_test_images(yolo_cfg["data_yaml"], split="test", num_images=5)

        preds_boxes, preds_labels = [], []
        results = yolo_model.val(data=yolo_cfg["data_yaml"], split="test", save_dir=None)
        with torch.no_grad():
            for i, img in enumerate(test_images):
                if i >= 5:
                    break
                pred = yolo_model.predict(img)
                boxes = pred[0].boxes.xyxy.cpu()
                labels = pred[0].boxes.cls.cpu()
                preds_boxes.append(boxes)
                preds_labels.append(labels)

        save_qualitative_images(
            test_images, preds_boxes, preds_labels,
            gts_boxes, gts_labels,
            model_name="YOLO", dataset_name=dataset_name,
            folder_path=qualitative_folder
        )

        print(f"\nFinished training on {dataset_name}.\n")


    # ==========================
    # Final Summary Table
    # ==========================
    summary_df = pd.DataFrame(summary_rows)

    ordered_cols = [
        "Dataset",
        "Model",
        "mAP50",
        "mAP50_95",
        "Precision",
        "Recall",
        "Inference Speed"
    ]

    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]
    summary_df = summary_df.round(4)

    print("\n=== Final Test Metrics Summary ===")
    print(summary_df.to_string(index=False))
