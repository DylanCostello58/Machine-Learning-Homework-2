import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import resize_image_and_boxes, IMAGE_SIZE


class DetectionDataset(Dataset):
    def __init__(self, root, data_list, resize_dim=IMAGE_SIZE):
        """
        Args:
            root (str): Project root folder (not joined to image paths anymore)
            data_list (list): list of dicts with keys: "image", "boxes", "labels"
            resize_dim (int): target image size (square)
        """
        self.root = root  # kept for reference, not used to join paths
        self.data = data_list
        self.resize_dim = resize_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # JSON already contains the correct image path, no need to join root
        img_path = item["image"]

        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(item["boxes"], dtype=torch.float32)
        labels = torch.tensor(item["labels"], dtype=torch.int64)

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        image, boxes = resize_image_and_boxes(image, boxes, self.resize_dim)

        return image, {"boxes": boxes, "labels": labels}
