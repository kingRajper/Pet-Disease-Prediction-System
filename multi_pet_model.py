import torch.nn as nn
import torchvision.models as models

class MultiPetDiseaseModel(nn.Module):
    def __init__(self, num_diseases):
        super(MultiPetDiseaseModel, self).__init__()

        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2], nn.AdaptiveAvgPool2d(1))

        self.species_classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.disease_classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout to prevent overfitting
            nn.Linear(512, num_diseases)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.flatten(start_dim=1)
        species_output = self.species_classifier(features)
        disease_output = self.disease_classifier(features)
        return species_output, disease_output
