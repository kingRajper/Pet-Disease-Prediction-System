import pandas as pd
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class PetDiseaseDataset(Dataset):
    def __init__(self, texts, species_labels, disease_labels):
        self.encodings = tokenizer(list(map(str, texts)), truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.species_labels = torch.tensor(species_labels, dtype=torch.long)
        self.disease_labels = torch.tensor(disease_labels, dtype=torch.long)

    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["species_label"] = self.species_labels[idx]
        item["disease_label"] = self.disease_labels[idx]
        return item

def load_dataset(csv_path):
    """Load and preprocess dataset."""
    df = pd.read_csv(csv_path)
    
    # Ensure all text entries are strings
    df["Symptoms"] = df["Symptoms"].astype(str).fillna("No description available")
    
    # Convert species and disease labels to categorical values
    species_mapping = {species: idx for idx, species in enumerate(df["Species"].unique())}
    disease_mapping = {disease: idx for idx, disease in enumerate(df["Disease"].unique())}

    print("🔍 Training Species Mapping:", species_mapping)
    print("🔍 Training Disease Mapping:", disease_mapping)

    
    df["species_label"] = df["Species"].map(species_mapping)
    df["disease_label"] = df["Disease"].map(disease_mapping)
    
    # Split dataset into training and test sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_dataset = PetDiseaseDataset(train_df["Symptoms"].tolist(), train_df["species_label"].tolist(), train_df["disease_label"].tolist())
    test_dataset = PetDiseaseDataset(test_df["Symptoms"].tolist(), test_df["species_label"].tolist(), test_df["disease_label"].tolist())
    
    return train_dataset, test_dataset, species_mapping, disease_mapping