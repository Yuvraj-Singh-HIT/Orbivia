print("TRAINING SCRIPT STARTED")

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.utils.dataset import SegmentationDataset
from backend.ml.models.segformer import get_model
from backend.utils.metrics import iou_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

train_ds = SegmentationDataset(
    img_dir=os.path.join(BASE_DIR, "datasets", "dataset", "train", "images"),
    mask_dir=os.path.join(BASE_DIR, "datasets", "dataset", "train", "masks"),
    size=cfg["image_size"],
)

val_ds = SegmentationDataset(
    img_dir=os.path.join(BASE_DIR, "datasets", "dataset", "val", "images"),
    mask_dir=os.path.join(BASE_DIR, "datasets", "dataset", "val", "masks"),
    size=cfg["image_size"],
)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

if len(train_ds) == 0 or len(val_ds) == 0:
    raise RuntimeError("Dataset empty! Check dataset paths.")

train_dl = DataLoader(
    train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
)

val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = get_model(cfg["num_classes"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
criterion = nn.CrossEntropyLoss()

for epoch in range(cfg["epochs"]):
    model.train()
    total_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

    for imgs, masks in tqdm(train_dl, desc="Training"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dl)

    model.eval()
    total_iou = 0.0

    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            total_iou += iou_score(preds, masks, cfg["num_classes"]).item()

    avg_iou = total_iou / len(val_dl)

    print(f"Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f}")

torch.save(model.state_dict(), os.path.join(BASE_DIR, "weights", "segformer.pth"))
print("Model saved to weights/segformer.pth")
