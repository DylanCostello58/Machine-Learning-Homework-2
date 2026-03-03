# trainRCNN.py

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import time
from modelRCNN import get_model
from utils import evaluate_detections, collate_fn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_model(
    train_dataset,
    val_dataset,
    test_dataset=None,
    num_classes=2,
    epochs=30,
    patience=5,
    checkpoint_prefix="model",
    batch_size_train=2,
    batch_size_val=1,
    learning_rate=0.005
):
    # -------------------------
    # Data loaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        collate_fn=collate_fn
    )

    # -------------------------
    # Model and optimizer
    # -------------------------
    model = get_model(num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    # -------------------------
    # Transformation for RCNN
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # resize all images
        transforms.ToTensor()
    ])

    best_map = 0.0
    epochs_without_improve = 0
    total_training_time = 0.0
    final_metrics = {
        "mAP50": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "training_time": 0.0,
        "Inference Speed": 0.0
    }
    test_metrics = None

    print("\nStarting training...\n")

    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        start_train = time.time()
        total_loss = 0.0

        for images, targets in train_loader:
            images = [transform(img).to(device) for img in images]  # resize images
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        training_time = time.time() - start_train
        total_training_time += training_time

        # -------------------------
        # Validation evaluation
        # -------------------------
        model.eval()
        preds, gts = [], []
        total_images = 0
        start_infer = time.time()
        with torch.no_grad():
            for images, targets in val_loader:
                images_tensor = [transform(img).to(device) for img in images]
                outputs = model(images_tensor)

                for i in range(len(outputs)):
                    scores = outputs[i]['scores'].cpu()
                    keep = scores > 0.5
                    pred_boxes = outputs[i]['boxes'][keep].cpu()
                    pred_labels = outputs[i]['labels'][keep].cpu()

                    preds.append((pred_boxes, pred_labels))
                    gts.append((targets[i]['boxes'], targets[i]['labels']))
                    total_images += 1

        inference_time = time.time() - start_infer
        inference_speed = total_images / (inference_time + 1e-6)
        mAP50, precision, recall = evaluate_detections(preds, gts, num_classes=num_classes)

        final_metrics = {
            "mAP50": mAP50,
            "Precision": precision,
            "Recall": recall,
            "training_time": total_training_time,
            "Inference Speed": inference_speed
        }

        # -------------------------
        # Early stopping and logging
        # -------------------------
        if mAP50 > best_map:
            best_map = mAP50
            epochs_without_improve = 0
            torch.save(model.state_dict(), f"{checkpoint_prefix}_best.pth")
            print("New best model saved.")
        else:
            epochs_without_improve += 1

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"Loss: {total_loss:.4f}")
        print(f"mAP@0.5: {mAP50:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Training time: {training_time:.2f}s")
        print(f"Inference speed: {final_metrics['Inference Speed']:.2f} images/sec")
        print(f"Best mAP@0.5 so far: {best_map:.4f}")
        print(f"Epochs without improvement: {epochs_without_improve}/{patience}")

        if epochs_without_improve >= patience:
            print("\nEarly stopping triggered.")
            break

    # -------------------------
    # Test evaluation (optional)
    # -------------------------
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn)
        model.eval()
        preds, gts = [], []
        total_images = 0
        start_infer = time.time()
        with torch.no_grad():
            for images, targets in test_loader:
                images_tensor = [transform(img).to(device) for img in images]
                outputs = model(images_tensor)

                for i in range(len(outputs)):
                    scores = outputs[i]['scores'].cpu()
                    keep = scores > 0.5
                    pred_boxes = outputs[i]['boxes'][keep].cpu()
                    pred_labels = outputs[i]['labels'][keep].cpu()

                    preds.append((pred_boxes, pred_labels))
                    gts.append((targets[i]['boxes'], targets[i]['labels']))
                    total_images += 1

        inference_time = time.time() - start_infer
        inference_speed = total_images / (inference_time + 1e-6)
        mAP50, precision, recall = evaluate_detections(preds, gts, num_classes=num_classes)

        test_metrics = {
            "mAP50": mAP50,
            "Precision": precision,
            "Recall": recall,
            "Inference Speed": inference_speed
        }

        print("\nTest set evaluation:")
        print(f"mAP@0.5: {mAP50:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Inference speed: {test_metrics['Inference Speed']:.2f} images/sec")

    print("\nTraining complete.")
    print(f"Best mAP@0.5 achieved: {best_map:.4f}")

    return final_metrics, test_metrics
