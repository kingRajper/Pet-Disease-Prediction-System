from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from text_model import PetDiseaseTextClassifier
from multi_pet_model import MultiPetDiseaseModel
from image_dataset_loader import get_disease_classes
import os

# Initialize app
app = FastAPI()

# ---------- TEXT MODEL SETUP ----------
text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = PetDiseaseTextClassifier(num_species=3, num_diseases=16)
text_model.load_state_dict(torch.load("text_disease_model.pth", map_location="cpu"))
text_model.eval()

text_species_mapping = {0: "Cat", 1: "Dog", 2: "Fish"}
text_disease_mapping = {
    0: "cat_ringworm", 1: "cat_scabies", 2: "dermatitis", 3: "fine", 4: "flea_allergy",
    5: "Dog_Ringworm", 6: "Dog_Scabies", 7: "Healthy_Dog", 8: "Hotspot",
    9: "Aeromoniasis Bacterial diseases", 10: "Bacterial disease gill",
    11: "Bacterial Red disease", 12: "Fungal Saprolegniasis diseases",
    13: "Healthy Fish", 14: "Parasitic diseases", 15: "Viral White disease diseases tail"
}

# ---------- IMAGE MODEL SETUP ----------
data_dir = "dataset"
disease_class_mapping = get_disease_classes(os.path.join(data_dir, "train"))
image_disease_mapping = [k for k, _ in sorted(disease_class_mapping.items(), key=lambda x: x[1])]
image_species_mapping = {0: "Dog", 1: "Cat", 2: "Fish"}

image_model = MultiPetDiseaseModel(num_diseases=len(image_disease_mapping))
image_model.load_state_dict(torch.load("best_multi_pet_disease_model.pth", map_location="cpu"))
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- REQUEST MODELS ----------
class TextInput(BaseModel):
    text: str

# ---------- TEXT ENDPOINT ----------
@app.post("/predict-text/")
def predict_text(payload: TextInput):
    encoding = text_tokenizer(payload.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    with torch.no_grad():
        species_output, disease_output = text_model(input_ids, attention_mask)
        species_probs = F.softmax(species_output, dim=1)
        disease_probs = F.softmax(disease_output, dim=1)
        species_pred = species_probs.argmax(dim=1).item()
        disease_pred = disease_probs.argmax(dim=1).item()

    return {
        "species": text_species_mapping[species_pred],
        "species_confidence": round(species_probs[0][species_pred].item(), 4),
        "disease": text_disease_mapping[disease_pred],
        "disease_confidence": round(disease_probs[0][disease_pred].item(), 4)
    }

# ---------- IMAGE ENDPOINT ----------
@app.post("/predict-image/")
def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        species_output, disease_output = image_model(image_tensor)
        species_probs = F.softmax(species_output, dim=1)
        disease_probs = F.softmax(disease_output, dim=1)
        species_pred = species_probs.argmax(dim=1).item()
        disease_pred = disease_probs.argmax(dim=1).item()

    return {
        "species": image_species_mapping[species_pred],
        "species_confidence": round(species_probs[0][species_pred].item(), 4),
        "disease": image_disease_mapping[disease_pred],
        "disease_confidence": round(disease_probs[0][disease_pred].item(), 4)
    }
