# utils.py

import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import yaml
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

# -----------------------------
# Dataset info
# -----------------------------
PET_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British_Shorthair", "Egyptian_Mau", "Maine_Coon",
    "Persian", "Ragdoll", "Siamese"
]

BREED_LABEL_MAP = {b: i+1 for i, b in enumerate(PET_BREEDS)}  # RCNN uses 1-based labels

# -----------------------------
# Data utilities
# -----------------------------
def collate_fn(batch):
    return tuple(zip(*batch))

def resize_image_and_boxes(image, boxes, target_size):
    w, h = image.size
    image = image.resize((target_size, target_size))
    scale_x = target_size / w
    scale_y = target_size / h
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return image, boxes

def parse_pet_xml(xml_path, breed_label):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(breed_label)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

def masks_to_boxes(mask_path, start_label=1):
    """
    Converts a segmentation mask to bounding boxes.
    start_label=1 for RCNN (1 = first object class)
    start_label=0 for YOLO (0 = first object class)
    """
    mask = np.array(Image.open(mask_path))
    boxes = []
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        ys, xs = np.where(mask == obj_id)
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    labels = torch.full((len(boxes),), start_label, dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), labels

# -----------------------------
# Metrics
# -----------------------------
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-6
    return inter / union

def evaluate_detections(preds, gts, iou_thresh=0.5, num_classes=None):
    """
    Computes:
        - mAP50: mean of per-class precision @ IoU>=0.5
        - precision: micro-average over all boxes
        - recall: micro-average over all boxes
    """
    if num_classes is None:
        all_labels = torch.cat([labels for _, labels in gts], dim=0)
        num_classes = int(all_labels.max().item())

    per_class_AP = []
    total_TP, total_FP, total_FN = 0, 0, 0

    for cls in range(1, num_classes + 1):
        TP, FP, FN = 0, 0, 0
        for (p_boxes, p_labels), (g_boxes, g_labels) in zip(preds, gts):
            p_cls = p_boxes[p_labels == cls]
            g_cls = g_boxes[g_labels == cls]
            matched = set()
            for pb in p_cls:
                found = False
                for i, gb in enumerate(g_cls):
                    if i in matched:
                        continue
                    if box_iou(pb.tolist(), gb.tolist()) >= iou_thresh:
                        TP += 1
                        matched.add(i)
                        found = True
                        break
                if not found:
                    FP += 1
            FN += len(g_cls) - len(matched)

        precision_cls = TP / (TP + FP + 1e-6) if (TP + FP) > 0 else 0.0
        per_class_AP.append(precision_cls)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    mAP50 = np.mean(per_class_AP)
    precision = total_TP / (total_TP + total_FP + 1e-6)
    recall = total_TP / (total_TP + total_FN + 1e-6)

    return mAP50, precision, recall

# -----------------------------
# YOLO utilities
# -----------------------------
def convert_boxes_to_yolo(boxes, img_w, img_h):
    """
    Converts [x1, y1, x2, y2] boxes to YOLO format [x_center, y_center, w, h]
    """
    yolo_boxes = []
    for b in boxes:
        x_center = (b[0] + b[2]) / 2 / img_w
        y_center = (b[1] + b[3]) / 2 / img_h
        w = (b[2] - b[0]) / img_w
        h = (b[3] - b[1]) / img_h
        yolo_boxes.append([x_center, y_center, w, h])
    return yolo_boxes

# -----------------------------
# YOLO test image loader
# -----------------------------
import yaml

def load_yolo_test_images(data_yaml_path, split="test", num_images=5):
    """
    Loads images and ground truth boxes/labels from a YOLO dataset for qualitative visualization.

    Returns:
        images: list of PIL.Image
        boxes: list of torch.Tensor, each [N,4] in xyxy format
        labels: list of torch.Tensor, each [N]
    """
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if split not in data:
        raise ValueError(f"{split} split not found in {data_yaml_path}")

    img_paths = data[split]
    images, boxes_list, labels_list = [], [], []

    for img_path in img_paths[:num_images]:
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        w, h = img.size

        # Assume labels are in same folder structure as YOLO: replace /images/ with /labels/ and .jpg → .txt
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        try:
            with open(label_path, "r") as f:
                bboxes, lbls = [], []
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x_c, y_c, bw, bh = map(float, parts)
                    # Convert YOLO format [x_center, y_center, w, h] -> xyxy
                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h
                    x2 = (x_c + bw / 2) * w
                    y2 = (y_c + bh / 2) * h
                    bboxes.append([x1, y1, x2, y2])
                    lbls.append(int(cls))
                boxes_list.append(torch.tensor(bboxes, dtype=torch.float32))
                labels_list.append(torch.tensor(lbls, dtype=torch.int64))
        except FileNotFoundError:
            # If no label file, just add empty tensors
            boxes_list.append(torch.zeros((0, 4), dtype=torch.float32))
            labels_list.append(torch.zeros((0,), dtype=torch.int64))

    return images, boxes_list, labels_list

def load_yolo_test_images(yaml_path, split="test", num_images=5):
    """
    Loads YOLO test images and their ground truth boxes and labels.
    Returns:
        images: list of PIL Images
        boxes: list of torch.FloatTensor [N,4] in [x1,y1,x2,y2]
        labels: list of torch.LongTensor [N]
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    img_folder = data[split]  # e.g., "images/test"
    label_folder = os.path.join(os.path.dirname(img_folder), "labels")  # assumes labels are in parallel folder

    img_paths = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))[:num_images]

    images = []
    boxes_list = []
    labels_list = []

    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_folder, f"{img_name}.txt")

        img = Image.open(img_path).convert("RGB")
        images.append(img)

        if os.path.exists(label_path):
            lbls = []
            bxs = []
            w, h = img.size
            with open(label_path, "r") as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    x1 = (x_c - bw/2) * w
                    y1 = (y_c - bh/2) * h
                    x2 = (x_c + bw/2) * w
                    y2 = (y_c + bh/2) * h
                    bxs.append([x1, y1, x2, y2])
                    lbls.append(int(cls))
            boxes_list.append(torch.tensor(bxs, dtype=torch.float32))
            labels_list.append(torch.tensor(lbls, dtype=torch.int64))
        else:
            boxes_list.append(torch.empty((0,4), dtype=torch.float32))
            labels_list.append(torch.empty((0,), dtype=torch.int64))

    return images, boxes_list, labels_list
