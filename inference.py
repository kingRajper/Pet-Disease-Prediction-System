import torch
import torchvision.transforms as transforms
from PIL import Image
from multi_pet_model import MultiPetDiseaseModel
from image_dataset_loader import get_disease_classes
import os

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load disease class names dynamically
data_dir = "dataset"
disease_classes = list(get_disease_classes(os.path.join(data_dir, "train")).keys())

# Define species class names
species_classes = ["Dog", "Cat", "Fish"]

# Load trained model
model = MultiPetDiseaseModel(num_diseases=len(disease_classes)).to(device)
model.load_state_dict(torch.load("best_multi_pet_disease_model.pth", map_location=device))
model.eval()

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    """Predict species and disease from an image file."""
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        species_output, disease_output = model(image)
        species_pred = species_output.argmax(1).item()
        disease_pred = disease_output.argmax(1).item()

    print(f"✅ Predicted Species: {species_classes[species_pred]}")
    print(f"✅ Predicted Disease: {disease_classes[disease_pred]}")

# Example usage
if __name__ == "__main__":
    predict("preprocessed_RINGWORM_014_jpeg.rf.8f245571c272634d6204a21c220a6da2.jpg")  # Replace with actual image path
