import os
import cv2
import torch
import numpy as np
import json
from io import BytesIO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from backend.ml.models.segformer import get_model
from backend.utils.dataset import SegmentationDataset
from torch.utils.data import DataLoader
import yaml


def load_model_and_config():
    cfg = yaml.safe_load(
        open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["num_classes"]).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "weights", "segformer.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, cfg, device
    return None, cfg, device


def evaluate_model():
    model, cfg, device = load_model_and_config()
    if model is None:
        return {"error": "Model not found. Please train the model first."}

    val_ds = SegmentationDataset(
        img_dir=os.path.join(
            os.path.dirname(__file__), "datasets", "dataset", "val", "images"
        ),
        mask_dir=os.path.join(
            os.path.dirname(__file__), "datasets", "dataset", "val", "masks"
        ),
        size=cfg["image_size"],
    )

    val_dl = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )

    all_preds = []
    all_targets = []
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)

            pred_labels = torch.argmax(preds, dim=1)
            all_preds.extend(pred_labels.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

            for cls in range(cfg["num_classes"]):
                pred_cls = pred_labels == cls
                target_cls = masks == cls
                intersection = (pred_cls & target_cls).sum().float()
                union = (pred_cls | target_cls).sum().float()
                if union != 0:
                    total_iou += (intersection / union).item()

            total_samples += imgs.size(0)

    avg_iou = total_iou / (cfg["num_classes"] * total_samples)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    cm = confusion_matrix(
        all_targets, all_preds, labels=list(range(cfg["num_classes"]))
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Trees",
            "Lush Bushes",
            "Dry Grass",
            "Dry Bushes",
            "Ground Clutter",
            "Flowers",
            "Logs",
            "Rocks",
            "Landscape",
            "Sky",
        ],
        yticklabels=[
            "Trees",
            "Lush Bushes",
            "Dry Grass",
            "Dry Bushes",
            "Ground Clutter",
            "Flowers",
            "Logs",
            "Rocks",
            "Landscape",
            "Sky",
        ],
    )
    plt.title("Confusion Matrix - Segformer Model")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    cm_buffer = BytesIO()
    plt.savefig(cm_buffer, format="png", dpi=150, bbox_inches="tight")
    cm_buffer.seek(0)
    cm_image = base64.b64encode(cm_buffer.getvalue()).decode("utf-8")
    plt.close()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "iou_score": float(avg_iou),
        "confusion_matrix_image": cm_image,
        "total_samples": total_samples,
        "num_classes": cfg["num_classes"],
    }

    return metrics


def create_metrics_visualization(metrics):
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "IoU Score"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["iou_score"],
    ]

    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = plt.bar(metric_names, metric_values, color=colors)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plt.close()

    return buffer


def check_model_status():
    model_path = os.path.join(os.path.dirname(__file__), "weights", "segformer.pth")
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        return {
            "trained": True,
            "model_file": "segformer.pth",
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "message": "Model is already trained and ready for inference.",
        }
    return {"trained": False, "message": "Model not found. Please run training script."}
