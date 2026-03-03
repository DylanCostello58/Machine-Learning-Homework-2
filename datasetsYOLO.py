import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from utils import IMAGE_SIZE, convert_boxes_to_yolo

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._load_labels(label_path, image.size)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels

    def _load_labels(self, label_path, img_size):
        w, h = img_size
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:])
                    # Convert YOLO normalized back to absolute boxes
                    x1 = (x_center - bw/2) * w
                    y1 = (y_center - bh/2) * h
                    x2 = (x_center + bw/2) * w
                    y2 = (y_center + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
