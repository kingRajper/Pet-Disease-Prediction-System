import torch
import torch.nn as nn
from transformers import DistilBertModel

class PetDiseaseTextClassifier(nn.Module):
    def __init__(self, num_species, num_diseases):
        super(PetDiseaseTextClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        
        # Classifier heads
        self.species_classifier = nn.Linear(self.bert.config.hidden_size, num_species)
        self.disease_classifier = nn.Linear(self.bert.config.hidden_size, num_diseases)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Take CLS token representation
        pooled_output = self.dropout(pooled_output)
        
        species_output = self.species_classifier(pooled_output)
        disease_output = self.disease_classifier(pooled_output)
        
        return species_output, disease_output
