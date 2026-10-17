---
title: Glastoma AI — Skin Cancer Detection
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🩺 Glastoma AI
### Intelligent Skin Cancer Detection & Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-94.7%25-success.svg?style=flat&logo=target)](https://github.com/princedarkness369/Glastoma-AI-Skin-Cancer-Detection-)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Tracked-orange.svg?logo=git-lfs&logoColor=white)](https://git-lfs.github.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20العربية-blueviolet.svg)](#features)

<p align="center">
  <b>Glastoma AI</b> is a state-of-the-art dermatological AI screening platform designed to assist clinicians and patients with rapid, accessible, and highly accurate preliminary analysis of skin lesions.
</p>

[Key Features](#-key-features) •
[Model Architecture](#-model-architecture) •
[Supported Lesions](#-supported-lesion-classes) •
[Quick Start](#-quick-start) •
[API Reference](#-api-endpoints) •
[Medical Disclaimer](#-medical-disclaimer)

---

</div>

## 🌟 Key Features

- **🎯 94.7% Ensemble Accuracy**: Powered by a multi-model deep learning ensemble trained on over 10,000 clinically validated dermoscopic images (HAM10000).
- **🌍 Bilingual & RTL Native**: Complete Arabic (العربية) and English user interface with seamless bidirectional typography and medical terminology.
- **📷 Dual Input Modes**: Live camera dermatoscopy capture with instant snapshot preview or direct high-resolution image upload.
- **🔬 Multi-Metric Feature Analysis**: Evaluates lesions against ABCDE dermatological criteria (Asymmetry, Border irregularity, Color variation, Diameter, Evolution, and Texture).
- **📊 Top-3 Differential Probabilities**: Displays the primary diagnosis along with ranked candidate probabilities and confidence distributions.
- **💡 Actionable Guidance & Triage**: Delivers localized medical explanations, recommended clinical exams, potential treatments, and urgency flags (Low, Medium, High/Urgent).
- **📁 Local Scan History**: Saves previous assessments locally in your browser with timestamped history and instant recall.
- **⚡ Fast Inference**: Optimized PyTorch model inference serving sub-second analysis via an asynchronous FastAPI backend.

---

## 🧠 Model Architecture

The core classifier uses a **Heavy Ensemble Architecture** that aggregates features from three complementary Convolutional Neural Network (CNN) backbones:

```
                      ┌────────────────────────┐
                      │   Input Lesion Image   │
                      │       (224 x 224)      │
                      └───────────┬────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
 ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
 │   ResNet-50   │        │  DenseNet-121 │        │EfficientNet-B3│
 └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │  Softmax Probability   │
                      │       Averaging        │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │  7-Class Classification│
                      │   (94.7% Ensemble Acc) │
                      └────────────────────────┘
```

---

## 📋 Supported Lesion Classes

The model classifies skin lesions into 7 diagnostic categories:

| Code | Disease Name | الاسم العربي | Risk Level | Description |
|:---:|:---|:---|:---:|:---|
| **mel** | **Melanoma** | الورم الميلانيني | 🚨 **High** | Most aggressive skin cancer; early detection is life-saving. |
| **bcc** | **Basal Cell Carcinoma** | سرطان الخلايا القاعدية | ⚠️ **Medium** | Common slow-growing skin cancer, rarely metastasizes. |
| **akiec** | **Actinic Keratoses** | تقران أكتيني | ⚠️ **Medium** | Pre-cancerous scaly patches caused by chronic sun damage. |
| **nv** | **Melanocytic Nevi** | شامات ميلانينية | 🟢 **Low** | Typical benign moles formed by clusters of melanocytes. |
| **bkl** | **Benign Keratosis** | تقرن حميد | 🟢 **Low** | Non-cancerous skin growths common with aging (seborrheic). |
| **vasc** | **Vascular Lesions** | آفات وعائية | 🟢 **Low** | Benign blood vessel proliferation (angiomas, pyogenic granuloma). |
| **df** | **Dermatofibroma** | ورم ليفي جلدي | 🟢 **Low** | Harmless, firm fibrous nodules commonly found on extremities. |

---

## 🗂️ Project Structure

```bash
Glastoma/
├── backend/
│   ├── api.py                  # FastAPI server with model endpoints & static frontend mount
│   ├── heavy_model_best.pth    # PyTorch ensemble weights (166 MB, Git LFS)
│   └── requirements.txt        # Python backend dependencies
├── frontend/
│   ├── index.html              # Modern responsive bilingual UI
│   ├── styles.css              # Glassmorphic dark/light styling & RTL support
│   ├── script.js               # Application logic, inference handler & camera control
│   └── logo.png                # Glastoma logo asset
├── Dockerfile                  # Container definition for Docker & Hugging Face Spaces
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git ignore rules (cache, venv, temporary files)
├── .gitattributes              # Git LFS configuration for *.pth model files
├── deploy.sh                   # Local bash launcher
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### Option 1: Run Locally with Python

#### 1. Clone the repository (with Git LFS):
```bash
git clone https://github.com/princedarkness369/Glastoma-AI-Skin-Cancer-Detection-.git
cd Glastoma-AI-Skin-Cancer-Detection-
git lfs pull
```

#### 2. Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

#### 4. Launch the application:
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser. Both the web interface and API are served together.

---

### Option 2: Run with Docker

```bash
# Build the Docker image
docker build -t glastoma-ai .

# Run the container
docker run -p 7860:7860 glastoma-ai
```
Visit **[http://localhost:7860](http://localhost:7860)**.

---

## 📡 API Endpoints

The backend exposes an interactive OpenAPI specification available at `http://localhost:8000/docs`:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves the interactive web interface (`frontend/index.html`) |
| `POST` | `/predict` | Upload an image file (`multipart/form-data`) → Returns prediction & **Arabic** guidance |
| `POST` | `/predict/en` | Upload an image file (`multipart/form-data`) → Returns prediction & **English** guidance |
| `GET` | `/health` | Server status and verification of model weight loading |
| `GET` | `/classes` | List of supported lesion classes in English and Arabic |
| `GET` | `/info/{lesion_type}` | In-depth clinical notes, treatment guidelines, and exam criteria |
| `GET` | `/api` | Service overview and model metadata |

#### Example Prediction Request (`curl`):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@sample_lesion.jpg"
```

---

## 🛡️ Medical Disclaimer

> [!IMPORTANT]
> **Not a Substitute for Professional Medical Diagnosis:**  
> Glastoma AI is an artificial intelligence research and clinical decision-support tool. It is intended to assist medical professionals and raise awareness of dermatological conditions. It does **not** replace physical examination, dermoscopy by a certified dermatologist, or histological biopsy. Any suspicious or changing lesion should be evaluated immediately by a qualified healthcare provider.

---

## 📄 License & Attribution

- **Dataset**: Trained on the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (*Human Against Machine with 10,000 training images*).
- **Author**: [princedarkness369](https://github.com/princedarkness369)

- ## 📬 Contact

**NAME**

*   LinkedIn: https://www.linkedin.com/in/shahd-al-rabaya-58b8882b1/
*   GitHub: https://github.com/princedarkness369/
*   Email: Shahdfalrabaya@gmail.com

