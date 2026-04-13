# Offroad Semantic Scene Segmentation using Digital Twins

## Description
This repository contains an end-to-end implementation of a **semantic scene segmentation** model developed for the **Duality AI Offroad Autonomy Segmentation Hackathon**.

The project focuses on understanding off-road desert environments using **synthetic digital twin data** generated from **Duality AI’s Falcon simulation platform**.

The objective is to train a segmentation model on a synthetic desert environment and evaluate its performance on **unseen desert locations**, supporting real-world applications such as off-road autonomy, obstacle avoidance, and terrain understanding.

---

## Problem Statement
Off-road autonomous systems require fine-grained scene understanding to safely navigate complex and unstructured terrains.  
However, collecting and annotating large-scale real-world off-road data is expensive, time-consuming, and often impractical.

This project demonstrates how **digital twins and synthetic data** can be effectively used to train robust semantic segmentation models capable of handling **unseen environments**.

---

## Dataset
- **Source:** Duality AI Falcon Digital Twin Platform  
- **Type:** Fully synthetic desert environment data  

### Data Split
- **Train:** RGB images + segmentation masks  
- **Validation:** RGB images + segmentation masks  
- **Test:** RGB images only *(no ground truth used)*  

⚠️ **Test masks were not used during training or evaluation to prevent data leakage.**

---

## Segmentation Classes

| Class ID | Class Name |
|--------|-----------|
| 0 | Trees |
| 1 | Lush Bushes |
| 2 | Dry Grass |
| 3 | Dry Bushes |
| 4 | Ground Clutter |
| 5 | Flowers |
| 6 | Logs |
| 7 | Rocks |
| 8 | Landscape |
| 9 | Sky |

---

## Model Architecture
- **Model:** SegFormer (MIT-B0)  
- **Framework:** PyTorch  
- **Library:** segmentation-models-pytorch  
- **Input Resolution:** 256 × 256  
- **Training Device:** CPU  

A lightweight **SegFormer-B0** model was selected to ensure **stable training under CPU-only hardware constraints**.

---

## Evaluation Metric
Since this is a **semantic segmentation** task, model performance is evaluated using:

- **Mean Intersection-over-Union (mIoU)**

> Detection-based metrics such as mAP are not applicable to pixel-wise segmentation tasks.

**Validation mIoU:** `0.10`


---

## Project Structure
offroad_segmentation/
│
├── dataset/
│ ├── train/
│ │ ├── images/
│ │ └── masks/
│ ├── val/
│ │ ├── images/
│ │ └── masks/
│ └── test/
│ └── images/
│
├── models/
│ └── segformer.py
│
├── utils/
│ ├── dataset.py
│ └── metrics.py
│
├── outputs/ # Saved test predictions
├── train.py
├── test.py
├── config.yaml
├── requirements.txt
└── README.md
