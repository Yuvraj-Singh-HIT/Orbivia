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

**Orbivia** is an AI-powered terrain intelligence system built for **autonomous navigation** and **off-road traversability analysis**. Developed as part of the **Duality AI Offroad Autonomy Segmentation Hackathon**, Orbivia showcases how synthetic data generated from **digital twin environments** can train robust semantic segmentation models that generalize to unseen real-world terrain.

At its core, Orbivia uses a fine-tuned **SegFormer-B0** model to classify terrain pixels into 10 categories and provide actionable traversability decisions (GO / SLOW / STOP) — enabling safer autonomous vehicle operation in unstructured outdoor environments.

> 🏆 *Built for the Duality AI Falcon Digital Twin Platform Hackathon — using fully synthetic desert environment data for training.*

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

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Backend** | Flask (Python) |
| **ML Framework** | PyTorch |
| **Model Architecture** | SegFormer-B0 (Hugging Face Transformers) |
| **Visualization** | Chart.js, Matplotlib |
| **Data Storage** | JSON-based local database |
| **Dataset Source** | Duality AI Falcon Digital Twin Platform |

---

## 📁 Project Structure

```
offroad_segmentation/
├── app.py                          # Flask web application entry point
│
├── backend/
│   ├── ml/
│   │   ├── models/
│   │   │   └── segformer.py        # SegFormer model architecture & loading
│   │   ├── train.py                # Model training script
│   │   ├── evaluation.py           # Model evaluation logic
│   │   └── datasets/               # Training dataset files
│   │
│   ├── utils/
│   │   ├── dataset.py              # Dataset loading & preprocessing utilities
│   │   └── metrics.py              # mIoU, precision, recall, F1 computation
│   │
│   ├── services/
│   │   └── database.py             # Database read/write management
│   │
│   └── db/
│       └── database.json           # Local persistent storage (analysis logs)
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Main web interface (Jinja2 template)
│   └── static/
│       ├── css/
│       │   └── style.css           # Application styles
│       ├── js/
│       │   └── main.js             # Frontend logic & API calls
│       ├── favicon.svg             # Website favicon
│       └── results/                # Saved analysis output images
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python **3.8** or higher
- `pip` package manager
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

### Running the Application

```bash
# Start the Flask development server
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

### Usage

1. **Image Upload** — Click "Upload Image" and select a terrain photo to analyze.
2. **Video Analysis** — Upload a video file for frame-by-frame segmentation.
3. **Webcam Mode** — Enable your camera for live terrain detection.
4. **Dashboard** — View confusion matrices, class distributions, and traversability scores.

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Source** | Duality AI Falcon Digital Twin Platform |
| **Environment** | Fully synthetic desert terrain |
| **Type** | Annotated semantic segmentation masks |
| **Splits** | Train / Validation / Test — no data leakage |
| **Classes** | 10 terrain categories |

> The entire dataset was generated synthetically using Duality AI's photorealistic digital twin simulator, enabling diverse and annotated training data without manual labeling.

---

## 👥 Team

Orbivia was built collaboratively by a team of 4 developers, each owning a distinct area of the project:

---

### 🧑‍💻 Yuvraj Singh
**Role: ML Engineering & Model Architecture**

- Designed and implemented the **SegFormer-B0** model architecture (`backend/ml/models/segformer.py`)
- Led model training and hyperparameter tuning (`backend/ml/train.py`)
- Integrated PyTorch inference pipeline with the Flask backend
- Set up the overall **project repository** and managed version control

[![GitHub](https://img.shields.io/badge/GitHub-Yuvraj--Singh--HIT-181717?logo=github)](https://github.com/Yuvraj-Singh-HIT)

---

### 👩‍💻 Ashmita Ray
**Role: Backend Engineering & API Development**

- Developed the **Flask web application** and REST API routes (`app.py`)
- Built the **database service layer** for storing and retrieving analysis results (`backend/services/database.py`, `backend/db/database.json`)
- Implemented **video processing** and webcam integration in the backend
- Managed frontend–backend communication via JavaScript API calls (`frontend/static/js/main.js`)

[![GitHub](https://img.shields.io/badge/GitHub-AshCodeX025-181717?logo=github)](https://github.com/AshCodeX025)

---

### 👩‍💻 Shrabani Neogi
**Role: Dataset Pipeline & Evaluation**

- Handled **dataset preprocessing** and loading utilities (`backend/utils/dataset.py`)
- Implemented evaluation metrics — mIoU, Precision, Recall, F1 Score (`backend/utils/metrics.py`)
- Ran model **evaluation experiments** and generated performance reports (`backend/ml/evaluation.py`)
- Managed train/validation/test splits to ensure no data leakage

[![GitHub](https://img.shields.io/badge/GitHub-shrabani--stack-181717?logo=github)](https://github.com/shrabani-stack)

---

### 👩‍💻 Upasana Majumder
**Role: Frontend Development & UI/UX**

- Built the complete **web interface** (`frontend/templates/index.html`)
- Developed CSS styling and responsive layout (`frontend/static/css/style.css`)
- Created interactive **data visualizations** using Chart.js (confusion matrices, class distribution charts)
- Designed the **favicon** and overall visual identity of Orbivia

[![GitHub](https://img.shields.io/badge/GitHub-upasana23-181717?logo=github)](https://github.com/upasana23)

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
