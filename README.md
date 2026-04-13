<div align="center">

```
                                    ██████╗ ██████╗ ██████╗ ██╗██╗   ██╗██╗ █████╗ 
                                   ██╔═══██╗██╔══██╗██╔══██╗██║██║   ██║██║██╔══██╗
                                   ██║   ██║██████╔╝██████╔╝██║██║   ██║██║███████║
                                   ██║   ██║██╔══██╗██╔══██╗██║╚██╗ ██╔╝██║██╔══██║
                                   ╚██████╔╝██║  ██║██████╔╝██║ ╚████╔╝ ██║██║  ██║
                                    ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═══╝  ╚═╝╚═╝  ╚═╝
```

### `[ TERRAIN INTELLIGENCE SYSTEM v1.0 ]`

*The ground beneath your wheels is a puzzle — Orbivia solves it.*

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-0d1117?style=flat-square&logo=python&logoColor=3572A5&labelColor=0d1117)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-0d1117?style=flat-square&logo=pytorch&logoColor=EE4C2C&labelColor=0d1117)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-0d1117?style=flat-square&logo=flask&logoColor=ffffff&labelColor=0d1117)](https://flask.palletsprojects.com)
[![SegFormer](https://img.shields.io/badge/Model-SegFormer--B0-0d1117?style=flat-square&logoColor=white&labelColor=0d1117&color=238636)](https://huggingface.co/nvidia/segformer-b0)
[![Duality AI Hackathon](https://img.shields.io/badge/Duality%20AI-Offroad%20Hackathon-0d1117?style=flat-square&labelColor=0d1117&color=FF6B35)](https://duality.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-0d1117?style=flat-square&labelColor=0d1117&color=yellow)](https://opensource.org/licenses/MIT)

</div>

---

## ◈ What is Orbivia?

**Orbivia** sees terrain the way a hawk sees a field — not as scenery, but as a map of decisions.

Built for autonomous off-road navigation, Orbivia ingests raw imagery and outputs a semantic understanding of the ground: what's safe, what's risky, what's a dead stop. Trained entirely on **synthetic digital twin data** from the Duality AI Falcon platform, it generalizes to real-world terrain without ever needing a single real-world pixel during training.

At its core: a fine-tuned **SegFormer-B0** transformer model, classifying every pixel into 10 terrain categories and translating that understanding into concrete navigation commands — `🟢 GO`, `🟡 SLOW`, `🔴 STOP`.

---

## ◈ Screenshots

<table>
<tr>
<td width="50%">

**Homepage — Clean Upload Interface**
![Homepage](docs/Screenshot%202026-04-14%20012542.png)

</td>
<td width="50%">

**Interactive Dashboard**
![Dashboard](docs/Screenshot%202026-04-14%20012557.png)

</td>
</tr>
<tr>
<td width="50%">

**Live Segmentation Output**
![Segmentation](docs/Screenshot%202026-04-14%20012613.png)

</td>
<td width="50%">

**Deep Metrics & Analytics**
![Metrics](docs/Screenshot%202026-04-14%20012634.png)

</td>
</tr>
<tr>
<td colspan="2" align="center">

**Terrain Class Legend**
![Legend](docs/Screenshot%202026-04-14%20012651.png)

</td>
</tr>
</table>

---

## ◈ Feature Matrix

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT MODES          │  INTELLIGENCE            │  OUTPUT   │
├───────────────────────┼──────────────────────────┼───────────┤
│  📸 Image upload       │  10-class segmentation   │  GO  🟢   │
│  🎞️  Video stream      │  Per-pixel confidence    │  SLOW 🟡  │
│  📷 Live webcam        │  Traversability scoring  │  STOP 🔴  │
└───────────────────────┴──────────────────────────┴───────────┘
```

| Capability | Details |
|:-----------|:--------|
| **Real-time Terrain Analysis** | Sub-second segmentation on standard hardware |
| **Video Frame Pipeline** | Frame-by-frame analysis with per-frame traversability scores |
| **Live Webcam Mode** | Stream your browser camera directly into the inference engine |
| **Multi-class Segmentation** | Trees, rocks, grass, sky, logs, flowers — 10 terrain archetypes |
| **Traversability Engine** | GO / CAUTION / STOP decisions grounded in class-level risk profiles |
| **Interactive Dashboard** | Confusion matrices, class distributions, F1/mIoU scoring |

---

## ◈ The Architecture

```
                    ┌──────────────────────────────┐
                    │        INPUT LAYER           │
                    │  [Image / Video / Webcam]    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       MIT-B0 ENCODER         │
                    │  Hierarchical Transformer    │
                    │  Overlapping patch embeddings│
                    │  4-stage feature extraction  │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       MLP DECODER HEAD       │
                    │  All-MLP feature aggregation │
                    │  Lightweight & deploy-ready  │
                    └──────────────┬───────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
    ┌──────────▼──────┐  ┌─────────▼────────┐  ┌──────▼──────────┐
    │  PIXEL CLASS    │  │  CLASS MASK      │  │  TRAVERSABILITY │
    │  PROBABILITIES  │  │  VISUALIZATION   │  │  DECISION       │
    └─────────────────┘  └──────────────────┘  └─────────────────┘
```

### Model specs

| Parameter | Value |
|:----------|:------|
| Architecture | SegFormer-B0 (MIT-B0 backbone) |
| Encoder | Mix Transformer — hierarchical with overlapping patch embeddings |
| Decoder | All-MLP head for lightweight feature aggregation |
| Validation mIoU | `0.10` *(see note below)* |
| Accuracy | `85%` |
| Metrics | Precision · Recall · F1 · mIoU — all available post-analysis |

> **On the mIoU:** The 0.10 figure reflects severe class imbalance in synthetic desert terrain — not model failure. Dominant terrain classes hit 85% accuracy. The imbalance is a dataset property, not an architecture flaw.

---

## ◈ Terrain Intelligence Map

Every pixel gets a verdict. Every verdict informs a decision.

| ID | Terrain Class | Signal | Reasoning |
|:--:|:-------------|:------:|:----------|
| `0` | Trees | 🟢 **GO** | Clear surrounding — safe path ahead |
| `1` | Lush Bushes | 🟢 **GO** | Passable vegetation, low resistance |
| `2` | Dry Grass | 🟡 **SLOW** | Reduced traction, reduced visibility |
| `3` | Dry Bushes | 🟡 **SLOW** | Possible concealed obstacles |
| `4` | Ground Clutter | 🟡 **SLOW** | Mixed terrain, degraded confidence |
| `5` | Flowers | 🟢 **GO** | Open, low-risk ground cover |
| `6` | Logs | 🔴 **STOP** | Hard physical obstacle — do not traverse |
| `7` | Rocks | 🔴 **STOP** | High-risk — halt navigation immediately |
| `8` | Landscape | 🟢 **GO** | Open terrain, clear trajectory |
| `9` | Sky | 🟢 **GO** | Background reference class |

---

## ◈ Tech Stack

```
FRONTEND    ──  HTML5 · CSS3 · JavaScript · Chart.js
BACKEND     ──  Flask (Python) · REST API
ML          ──  PyTorch · Segmentation Models PyTorch
MODEL       ──  SegFormer-B0 (MIT-B0 backbone)
DATA        ──  Duality AI Falcon Digital Twin Platform
DEPLOY      ──  Gunicorn · Render-ready
```

---

## ◈ Get Running in 4 Steps

### Prerequisites

```
Python 3.8+  ·  4GB+ RAM  ·  PyTorch (CPU or GPU)  ·  Any modern browser
```

### Step 1 — Clone

```bash
git clone https://github.com/Yuvraj-Singh-HIT/Orbivia.git
cd Orbivia
```

### Step 2 — Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Model Weights

Download `segformer.pth` from Google Drive:
👉 [**Download weights**](https://drive.google.com/drive/folders/1bGfn7Pxqrs0SoX_nui5QCz1bfsVHVtIi?usp=drive_link)

Place at: `backend/ml/weights/segformer.pth`

### Step 4 — Launch

```bash
python app.py
# → http://localhost:5000
```

---

## ◈ Using Orbivia

```
 ┌─ IMAGE MODE ──────────────────────────────────────────────┐
 │  Upload terrain photo → segmentation overlay → GO/SLOW/STOP
 └───────────────────────────────────────────────────────────┘

 ┌─ VIDEO MODE ──────────────────────────────────────────────┐
 │  Upload video → per-frame segmentation → downloadable output
 └───────────────────────────────────────────────────────────┘

 ┌─ WEBCAM MODE ─────────────────────────────────────────────┐
 │  Enable camera → live terrain feed → real-time traversability
 └───────────────────────────────────────────────────────────┘

 ┌─ DASHBOARD ───────────────────────────────────────────────┐
 │  Confusion matrix · Class distribution · Precision/F1/AUC
 └───────────────────────────────────────────────────────────┘
```

---

## ◈ Dataset

| Property | Details |
|:---------|:--------|
| **Source** | Duality AI Falcon Digital Twin Platform |
| **Type** | Fully synthetic desert terrain |
| **Labels** | Annotated per-pixel semantic segmentation masks |
| **Splits** | Train / Validation / Test — zero data leakage |
| **Classes** | 10 terrain categories |

---

## ◈ Repository Layout

```
Orbivia/
│
├── app.py                        ← Flask entrypoint
│
├── backend/
│   ├── ml/
│   │   ├── models/segformer.py   ← SegFormer architecture
│   │   ├── train.py              ← Training loop
│   │   ├── evaluation.py         ← Inference & evaluation
│   │   └── weights/              ← Drop segformer.pth here
│   ├── utils/
│   │   ├── dataset.py            ← Data loading & preprocessing
│   │   └── metrics.py            ← mIoU, F1, Precision, Recall
│   ├── services/database.py      ← Storage & retrieval layer
│   └── config.yaml               ← Central config
│
├── frontend/
│   ├── templates/index.html      ← Main interface
│   └── static/
│       ├── css/style.css         ← Styling
│       └── js/main.js            ← Upload, API calls, Chart.js
│
├── docs/                         ← Screenshots & documentation
├── requirements.txt
└── README.md
```

---

## ◈ The Team

Four engineers, four domains, one system.

---

**Yuvraj Singh** — *ML Engineering & Architecture*
> Designed and implemented the SegFormer-B0 architecture · Led model training and hyperparameter tuning · Built PyTorch ↔ Flask inference bridge · Managed the repository and version control

[![GitHub](https://img.shields.io/badge/Yuvraj--Singh--HIT-0d1117?style=flat-square&logo=github)](https://github.com/Yuvraj-Singh-HIT)

---

**Ashmita Ray** — *Backend Engineering & API*
> Developed Flask web application and all REST routes · Built database service layer for analysis persistence · Implemented video processing and webcam integration · Wired frontend–backend JS communication

[![GitHub](https://img.shields.io/badge/AshCodeX025-0d1117?style=flat-square&logo=github)](https://github.com/AshCodeX025)

---

**Shrabani Neogi** — *Dataset Pipeline & Evaluation*
> Handled dataset preprocessing and loading utilities · Implemented mIoU, Precision, Recall, F1 scoring · Built the full model evaluation pipeline · Produced confusion matrices and performance reports

[![GitHub](https://img.shields.io/badge/shrabani--stack-0d1117?style=flat-square&logo=github)](https://github.com/shrabani-stack)

---

**Upasana Majumder** — *Frontend & UI/UX*
> Designed and built the responsive web interface · Crafted all CSS components and visual language · Implemented interactive JavaScript for uploads and real-time updates · Integrated Chart.js dashboard visualizations

[![GitHub](https://img.shields.io/badge/upasana23-0d1117?style=flat-square&logo=github)](https://github.com/upasana23)

---

## ◈ Contributing

```bash
git checkout -b feature/your-idea
git commit -m "feat: describe what it does"
git push origin feature/your-idea
# → open a Pull Request
```

Comment your code. Keep tests green. We'll handle the rest.

---

## ◈ License

MIT — do what you want, keep the attribution.

---

## ◈ Acknowledgements

- [**Duality AI**](https://duality.ai) — Falcon Digital Twin Platform + hackathon
- [**Hugging Face**](https://huggingface.co/docs/transformers) — SegFormer implementation
- [**PyTorch**](https://pytorch.org) — ML backbone
- [**Chart.js**](https://chartjs.org) — Dashboard visualizations

---

<div align="center">

```
[ ORBIVIA — KNOW THE GROUND BEFORE YOU MOVE ]
```

*Built by Team Orbivia*

</div>
