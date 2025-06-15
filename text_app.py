from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer
from text_model import PetDiseaseTextClassifier
import torch.nn.functional as F

# ✅ Define the request model
class TextInput(BaseModel):
    text: str

app = FastAPI()

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

species_mapping = {0: "Cat", 1: "Dog", 2: "Fish"}  # consistent with training
disease_mapping = {
    0: "cat_ringworm", 1: "cat_scabies", 2: "dermatitis", 3: "fine", 4: "flea_allergy",
    5: "Dog_Ringworm", 6: "Dog_Scabies", 7: "Healthy_Dog", 8: "Hotspot",
    9: "Aeromoniasis Bacterial diseases", 10: "Bacterial disease gill",
    11: "Bacterial Red disease", 12: "Fungal Saprolegniasis diseases",
    13: "Healthy Fish", 14: "Parasitic diseases", 15: "Viral White disease diseases tail"
}

model = PetDiseaseTextClassifier(num_species=len(species_mapping), num_diseases=len(disease_mapping))
model.load_state_dict(torch.load("text_disease_model.pth", map_location="cpu"))
model.eval()

@app.post("/predict-text/")
def predict_text(payload: TextInput):
    text = payload.text

    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        species_output, disease_output = model(input_ids, attention_mask)
        
        # Apply softmax to get probabilities
        species_probs = F.softmax(species_output, dim=1)
        disease_probs = F.softmax(disease_output, dim=1)

        species_pred = species_probs.argmax(dim=1).item()
        disease_pred = disease_probs.argmax(dim=1).item()

        return {
            "species": species_mapping[species_pred],
            "species_confidence": round(species_probs[0][species_pred].item(), 4),
            "disease": disease_mapping[disease_pred],
            "disease_confidence": round(disease_probs[0][disease_pred].item(), 4)
        }
