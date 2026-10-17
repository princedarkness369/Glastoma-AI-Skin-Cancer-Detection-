---
title: Glastoma AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Glastoma AI — Intelligent Skin Cancer Detection

Glastoma is an AI-assisted dermatology analysis system designed for clinical decision support. Trained on dermoscopic imaging to identify 7 types of skin lesions:

- **mel**: Melanoma (الورم الميلانيني)
- **bcc**: Basal Cell Carcinoma (سرطان الخلايا القاعدية)
- **akiec**: Actinic Keratoses (تقران أكتيني)
- **nv**: Melanocytic Nevi (شامات ميلانينية)
- **bkl**: Benign Keratosis (تقرن حميد)
- **vasc**: Vascular Lesions (آفات وعائية)
- **df**: Dermatofibroma (ورم ليفي جلدي)

## Architecture
- **Model**: Heavy Ensemble of ResNet-50, DenseNet-121, and EfficientNet-B3 (94.7% accuracy).
- **Backend**: FastAPI running on Python 3.10.
- **Frontend**: Responsive bilingual (EN/AR) interface with camera scanner and clinical metrics.
