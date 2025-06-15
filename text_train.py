import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from dataset_loader import load_dataset
from text_model import PetDiseaseTextClassifier
from collections import Counter

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset, test_dataset, species_mapping, disease_mapping = load_dataset("pet.csv")

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
model = PetDiseaseTextClassifier(num_species=len(species_mapping), num_diseases=len(disease_mapping)).to(device)

# Compute class weights for species and disease
species_labels = train_dataset.species_labels.tolist()
disease_labels = train_dataset.disease_labels.tolist()

species_counts = Counter(species_labels)
disease_counts = Counter(disease_labels)

species_weights = torch.tensor([
    len(species_labels) / species_counts[i] for i in range(len(species_mapping))
], dtype=torch.float).to(device)

disease_weights = torch.tensor([
    len(disease_labels) / disease_counts[i] for i in range(len(disease_mapping))
], dtype=torch.float).to(device)

# Apply weighted loss
species_criterion = nn.CrossEntropyLoss(weight=species_weights)
disease_criterion = nn.CrossEntropyLoss(weight=disease_weights)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_species = 0
    correct_disease = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        
        species_labels = batch["species_label"].to(device)
        disease_labels = batch["disease_label"].to(device)

        species_output, disease_output = model(input_ids, attention_mask)

        species_loss = species_criterion(species_output, species_labels)
        disease_loss = disease_criterion(disease_output, disease_labels)
        loss = species_loss + disease_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, species_preds = torch.max(species_output, 1)
        _, disease_preds = torch.max(disease_output, 1)

        correct_species += (species_preds == species_labels).sum().item()
        correct_disease += (disease_preds == disease_labels).sum().item()
        total_samples += species_labels.size(0)

    species_accuracy = 100 * correct_species / total_samples
    disease_accuracy = 100 * correct_disease / total_samples

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Species Accuracy: {species_accuracy:.2f}%, Disease Accuracy: {disease_accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "text_disease_model.pth")
print("✅ Model training complete. Saved as text_disease_model.pth")
