# 🐾 Pet Disease Prediction System

A multimodal AI system for predicting pet species and diseases using either symptom text or images. This project uses deep learning models (DistilBERT, EfficientNet) to classify across 3 species and 16 disease classes. Built for animal healthcare support and educational use.

---

## 📂 Project Structure

```
├── dataset/                     # Image dataset folder
│   ├── train/
│   │   ├── cat/
│   │   │   ├── cat_ringworm/
│   │   │   ├── cat_scabies/
│   │   │   ├── dermatits/
│   │   │   ├── fine/
│   │   │   └── flea_allergy/
│   │   ├── dog/
│   │   │   ├── Dog_Ringworm/
│   │   │   ├── Dog_Scabies/
│   │   │   ├── Helthy_Dog/
│   │   │   └── Hotspot/
│   │   └── fish/
│   │       ├── Aeromoniasis Bacterial diseases/
│   │       ├── Bacterial disease gill/
│   │       ├── Bacterial Red disease/
│   │       ├── Fungal Saprolegniasis diseases/
│   │       ├── Healthy Fish/
│   │       ├── Parasitic diseases/
│   │       └── Viral White disease diseases tail/
│   ├── val/
│   └── test/
├── multi_pet_model.py           # CNN-based Image model using EfficientNet
├── text_model.py               # Text classification model using DistilBERT
├── text_train.py               # Training script for text-based model
├── text_dataset_loader.py      # Dataset loader for text data
├── image_dataset_loader.py     # Dataset loader for image data (train/test/val splits)
├── image_train.py              # Training script for image model
├── main.py                     # Main FastAPI app with image and text endpoints
├── inference.py                # Inference utilities for image classification
├── text_app.py                 # FastAPI app for serving text model
├── pet.csv                     # Dataset with symptoms, species, diseases
├── requirements.txt            # Dependencies
└── README.md                   # You are here!
```

---

## 🧐 Features

- ✅ **Multitask Learning**: Predict both pet species and disease from text
- 📷 **Image Classifier**: Based on EfficientNet for vision-based disease detection
- 🤖 **NLP Classifier**: DistilBERT-based classifier for symptom text input
- ⚖️ **Imbalanced Dataset Handling**: Weighted loss for class balance
- ⚡ **FastAPI Inference**: Easily test predictions via `/predict-text` or `/predict-image` routes

---

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/pet-disease-predictor.git
cd pet-disease-predictor

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset

### 📝 Text Dataset

- Format: CSV (`pet.csv`)
- Columns:
  - `Symptoms` (text description of condition)
  - `Species` (Cat, Dog, Fish)
  - `Disease` (disease name, varies by species)
- Split: 80% training, 20% testing (done in `text_dataset_loader.py`)

### 🖼️ Image Dataset

- Directory Format:
  - `dataset/train/<species>/<disease>/`
  - `dataset/test/<species>/<disease>/`
  - `dataset/val/<species>/<disease>/`
- Total 3 species: `cat`, `dog`, `fish`
- Diseases under each species:
  - **Cat**: cat\_ringworm, cat\_scabies, dermatits, fine, flea\_allergy
  - **Dog**: Dog\_Ringworm, Dog\_Scabies, Helthy\_Dog, Hotspot
  - **Fish**: Aeromoniasis Bacterial diseases, Bacterial disease gill, Bacterial Red disease, Fungal Saprolegniasis diseases, Healthy Fish, Parasitic diseases, Viral White disease diseases tail

---

## 🧠 Image Dataset Loader

Your image data is managed using `image_dataset_loader.py`, which is responsible for:

- Reading images organized by species/disease
- Applying necessary transformations (resizing, normalization)
- Creating PyTorch `ImageFolder`-based loaders
- Returning `train_loader`, `val_loader`, `test_loader` for training EfficientNet

Example usage:

```python
from image_dataset_loader import get_image_dataloaders

train_loader, val_loader, test_loader = get_image_dataloaders(
    data_dir="dataset",
    image_size=224,
    batch_size=32
)
```

---

## 🏋️‍♀️ Train the Text Model

```bash
python text_train.py
```

- Outputs: `text_disease_model.pth`
- Logs training accuracy for species and disease prediction

---

## 🏋️‍♂️ Train the Image Model

```bash
python image_train.py
```

- Outputs: `image_disease_model.pth`
- Trains EfficientNet to classify species and diseases from images

---

## 🚀 Serve the Models with FastAPI

```bash
uvicorn main:app --reload
```

This uses `main.py` to expose both text and image endpoints.

### 🔤 Text Inference Endpoint

- Endpoint: `POST /predict-text/`
- Input JSON:

```json
{
  "text": "My dog has red spots and is itching constantly."
}
```

- Sample Output:

```json
{
  "species": "Dog",
  "species_confidence": 0.9821,
  "disease": "Dog_Scabies",
  "disease_confidence": 0.9447
}
```

### 🖼️ Image Inference Endpoint

- Endpoint: `POST /predict-image/`
- Input: Multipart image file
- Sample `curl` request:

```bash
curl -X POST http://localhost:8000/predict-image/ \
  -F image=@cat_flea.jpg
```

- Sample Output:

```json
{
  "species": "Cat",
  "species_confidence": 0.9712,
  "disease": "flea_allergy",
  "disease_confidence": 0.9315
}
```

---

## 🔮 To-Do

- ✅ Complete image training pipeline and app
- ✅ Integrate image inference via FastAPI
- 📈 Add evaluation metrics and confusion matrices
- 🧪 Add tests and validation scripts
- 📸 Add example input images and outputs in README

---

## 🤝 Contributors

- **Iqrar Ali** — AI Developer, ML Specialist, Deep Learning Enthusiast

---

## 📄 License

MIT License. See `LICENSE` file for details.

