<div align="center">

# 🛰️ Orbivia — AI Terrain Intelligence

**Autonomous off-road navigation powered by semantic segmentation and digital twins**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon](https://img.shields.io/badge/Duality%20AI-Offroad%20Hackathon-orange)](https://duality.ai)

</div>

---

## 📌 Overview

**Orbivia** is an AI-powered terrain intelligence system built for **autonomous navigation** and **off-road traversability analysis**. Orbivia showcases how synthetic data generated from **digital twin environments** can train robust semantic segmentation models that generalize to unseen real-world terrain.

At its core, Orbivia uses a fine-tuned **SegFormer-B0** model to classify terrain pixels into 10 categories and provide actionable traversability decisions (GO / SLOW / STOP) — enabling safer autonomous vehicle operation in unstructured outdoor environments.

---

## 🖼️ Demo Screenshots

### 1. Homepage
![Homepage](docs/Screenshot%202026-04-14%20012542.png)
Clean and intuitive interface for uploading terrain images and videos.

### 2. Interactive Dashboard - Upload Interface
![Results](docs/Screenshot%202026-04-14%20012557.png)
Visual segmentation output showing terrain classification with color-coded classes.

### 3. Real-time Segmentation Results
![Dashboard](docs/Screenshot%202026-04-14%20012613.png)
Comprehensive analytics with performance metrics and visualizations.

### 4. Detailed Analysis & Metrics
![Analysis](docs/Screenshot%202026-04-14%20012634.png)
In-depth analysis with confusion matrices, class distributions, and traversability scores.

### 5. Terrain Legend
![Video](docs/Screenshot%202026-04-14%20012651.png)

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🖼️ **Real-time Terrain Analysis** | Upload images for instant semantic segmentation output |
| 🎥 **Video Processing** | Frame-by-frame segmentation pipeline for video inputs |
| 📷 **Webcam Support** | Live terrain analysis directly from your browser camera |
| 🗺️ **Multi-class Segmentation** | Identifies 10 terrain classes (trees, rocks, sky, grass, etc.) |
| 🚦 **Traversability Assessment** | AI-powered GO / CAUTION / STOP terrain evaluation |
| 📊 **Interactive Dashboard** | Visual analytics with charts, confusion matrices, and metrics |
| 🔁 **Model Evaluation** | Precision, Recall, F1, and mIoU scoring available post-analysis |

---

## 🧠 Model & Architecture

Orbivia uses **SegFormer-B0** (MIT-B0 backbone), a lightweight transformer-based semantic segmentation architecture well-suited for embedded and real-time deployment.

- **Encoder:** Mix Transformer (MiT-B0) — hierarchical with overlapping patch embeddings
- **Decoder:** MLP decoder head for all-MLP feature aggregation
- **Input Resolution:** Configurable; optimized for efficiency
- **Output:** Per-pixel class probabilities across 10 terrain labels

### Model Performance

| Metric | Value |
|--------|-------|
| Validation mIoU | 0.10 |
| Accuracy | 85% |
| Precision / Recall / F1 | Available in dashboard post-analysis |

> *Note: mIoU is lower due to severe class imbalance in synthetic desert terrain; accuracy remains strong for dominant classes.*

---

## 🗺️ Segmentation Classes & Traversability

| Class ID | Class Name | Traversability | Meaning |
|----------|-----------|----------------|---------|
| 0 | Trees | 🟢 GO | Clear surroundings, safe to proceed |
| 1 | Lush Bushes | 🟢 GO | Passable vegetation |
| 2 | Dry Grass | 🟡 SLOW | Proceed with caution |
| 3 | Dry Bushes | 🟡 SLOW | Possible obstruction |
| 4 | Ground Clutter | 🟡 SLOW | Mixed terrain, reduced confidence |
| 5 | Flowers | 🟢 GO | Open, low-risk ground cover |
| 6 | Logs | 🔴 STOP | Physical obstacle, do not traverse |
| 7 | Rocks | 🔴 STOP | High-risk terrain, halt navigation |
| 8 | Landscape | 🟢 GO | Open terrain |
| 9 | Sky | 🟢 GO | Background reference class |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **Backend** | Flask (Python), REST API |
| **ML Framework** | PyTorch, Segmentation Models PyTorch |
| **Model** | SegFormer-B0 (MIT-B0 backbone) |
| **Data** | Duality AI Falcon Digital Twin Platform |
| **Deployment** | Gunicorn, Render-ready |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 4GB+ RAM
- PyTorch (CPU or GPU)
- A modern web browser

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yuvraj-Singh-HIT/Orbivia.git
cd Orbivia

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Model Weights

Download the trained model weights from Google Drive:
[**Download segformer.pth**](https://drive.google.com/drive/folders/1bGfn7Pxqrs0SoX_nui5QCz1bfsVHVtIi?usp=drive_link)

Place the file at: `backend/ml/weights/segformer.pth`

### Running the Application

```bash
# Start the Flask development server
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

## 📖 Usage Guide

### 1. Image Upload
Click "Upload Image" and select a terrain photo to analyze. The system will:
- Process the image through the segmentation model
- Display color-coded terrain classes
- Show traversability assessment (GO/SLOW/STOP)

### 2. Video Analysis
Upload a video file for frame-by-frame segmentation. The system processes each frame and provides:
- Real-time segmentation overlay
- Per-frame traversability scores
- Complete video with segmentation mask

### 3. Webcam Mode
Enable your camera for live terrain detection. Perfect for outdoor testing with a laptop or mobile device.

### 4. Dashboard
View comprehensive analytics including:
- Confusion matrix visualization
- Class distribution pie chart
- Performance metrics (Precision, Recall, F1, AUC)
- Traversability breakdown

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Source** | Duality AI Falcon Digital Twin Platform |
| **Environment** | Fully synthetic desert terrain |
| **Type** | Annotated semantic segmentation masks |
| **Splits** | Train / Validation / Test — no data leakage |
| **Classes** | 10 terrain categories |

---

## 🔧 Project Structure

```
Orbivia/
├── app.py                    # Main Flask application
├── backend/
│   ├── ml/
│   │   ├── models/           # SegFormer model implementation
│   │   ├── train.py          # Model training script
│   │   ├── evaluation.py     # Model evaluation
│   │   └── weights/          # Model weights (download separately)
│   ├── utils/                # Dataset & metrics utilities
│   ├── services/             # Database & API services
│   ├── config.yaml           # Configuration file
│   └── db/                   # Database JSON files
├── frontend/
│   ├── templates/            # HTML templates
│   └── static/               # CSS, JS, images
├── docs/                    # Documentation & screenshots
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 🤝 Contributing

We welcome contributions! To get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

Please make sure your code is well-commented and that existing tests pass before submitting.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Orbivia Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgements

- [**Duality AI**](https://duality.ai) — for the Falcon Digital Twin Platform and hackathon opportunity
- [**Hugging Face Transformers**](https://huggingface.co/docs/transformers) — for the SegFormer implementation
- [**PyTorch**](https://pytorch.org) — ML framework backbone
- [**Chart.js**](https://chartjs.org) — frontend data visualization

---

<div align="center">

Built with ❤️ by Team Orbivia

</div>
