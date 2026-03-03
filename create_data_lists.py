import os
import json
import random
import shutil
from PIL import Image
from utils import parse_pet_xml, masks_to_boxes, PET_BREEDS, BREED_LABEL_MAP

random.seed(42)  # deterministic splits
IMAGE_EXTS = [".jpg", ".png"]


# -----------------------------
# Helper Functions
# -----------------------------
def split_dataset(items):
    random.shuffle(items)
    n = len(items)
    return {
        "train": items[: int(0.7 * n)],
        "val": items[int(0.7 * n): int(0.85 * n)],
        "test": items[int(0.85 * n):],
    }


def convert_to_yolo(boxes, img_width, img_height):
    yolo_boxes = []
    for x1, y1, x2, y2 in boxes:
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        yolo_boxes.append([x_center, y_center, w, h])
    return yolo_boxes


def copy_and_write(split_items, split_name, out_dir):
    img_out = os.path.join(out_dir, "images", split_name)
    lbl_out = os.path.join(out_dir, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for item in split_items:
        src_img = item["image"]
        filename = os.path.basename(src_img)
        dst_img = os.path.join(img_out, filename)
        shutil.copy(src_img, dst_img)

        # write YOLO label
        img = Image.open(dst_img)
        w, h = img.size
        yolo_boxes = convert_to_yolo(item["boxes"], w, h)
        label_path = os.path.join(lbl_out, filename.rsplit(".", 1)[0] + ".txt")
        with open(label_path, "w") as f:
            for cls, box in zip(item["labels"], yolo_boxes):
                f.write(f"{cls} " + " ".join(map(str, box)) + "\n")


def save_yaml(yaml_path, out_dir, class_names):
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")


# -----------------------------
# Oxford-IIIT Pet
# -----------------------------
def create_pet_dataset():
    root = "Oxford-IIIT Pet"
    ann_dir = os.path.join(root, "Annotations", "xmls")
    img_dir = os.path.join(root, "images")
    json_out = os.path.join(root, "OxfordPetSubset_lists")
    yolo_out_dir = os.path.join(root, "YOLO_dataset")
    os.makedirs(json_out, exist_ok=True)
    os.makedirs(yolo_out_dir, exist_ok=True)

    items = []
    for xml_file in os.listdir(ann_dir):
        if not xml_file.endswith(".xml"):
            continue
        base = xml_file.replace(".xml", "")
        breed = base.rsplit("_", 1)[0]
        if breed not in PET_BREEDS:
            continue
        img_path = os.path.join(img_dir, base + ".jpg")
        if not os.path.exists(img_path):
            continue
        # RCNN labels 1-based
        boxes, labels = parse_pet_xml(os.path.join(ann_dir, xml_file), BREED_LABEL_MAP[breed])
        if boxes.numel() == 0:
            continue
        items.append({
            "image": img_path,
            "boxes": boxes.tolist(),
            "labels": labels.tolist()  # 1-based for RCNN
        })

    splits = split_dataset(items)

    # Save RCNN JSON
    for split, data in splits.items():
        with open(os.path.join(json_out, f"{split}.json"), "w") as f:
            json.dump(data, f)

    # Save YOLO dataset (labels converted to 0-based)
    for split_name, split_items in splits.items():
        for item in split_items:
            item["labels"] = [l - 1 for l in item["labels"]]  # convert 1->0
        copy_and_write(split_items, split_name, yolo_out_dir)

    # Save YAML next to this script
    save_yaml("data_oxford.yaml", yolo_out_dir, PET_BREEDS)

    print("\n===== Oxford-IIIT Pet =====")
    print(f"Total: {len(items)}, Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


# -----------------------------
# Penn-Fudan Pedestrian
# -----------------------------
def create_pennfudan_dataset():
    root = "PennFudanPed"
    img_dir = os.path.join(root, "PNGImages")
    mask_dir = os.path.join(root, "PedMasks")
    json_out = os.path.join(root, "PennFudan_lists")
    yolo_out_dir = os.path.join(root, "YOLO_dataset")
    os.makedirs(json_out, exist_ok=True)
    os.makedirs(yolo_out_dir, exist_ok=True)

    items = []
    for img_file in os.listdir(img_dir):
        if not img_file.endswith(".png"):
            continue
        mask_path = os.path.join(mask_dir, img_file.replace(".png", "_mask.png"))
        if not os.path.exists(mask_path):
            continue
        boxes, labels = masks_to_boxes(mask_path)
        if boxes.numel() == 0:
            continue
        items.append({
            "image": os.path.join(img_dir, img_file),
            "boxes": boxes.tolist(),
            "labels": labels.tolist()  # 1-based for RCNN, will convert for YOLO
        })

    splits = split_dataset(items)

    # Save RCNN JSON
    for split, data in splits.items():
        with open(os.path.join(json_out, f"{split}.json"), "w") as f:
            json.dump(data, f)

    # Save YOLO dataset (labels converted to 0-based)
    for split_name, split_items in splits.items():
        for item in split_items:
            item["labels"] = [0 for _ in item["labels"]]  # only 1 class
        copy_and_write(split_items, split_name, yolo_out_dir)

    # Save YAML next to this script
    save_yaml("data_fudan.yaml", yolo_out_dir, ["Pedestrian"])

    print("\n===== Penn-Fudan =====")
    print(f"Total: {len(items)}, Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    create_pet_dataset()
    create_pennfudan_dataset()
