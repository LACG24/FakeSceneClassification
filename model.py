import torch.nn as nn
from torchvision import models

# Clase personalizada para EfficientNet con Dropout
class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        # Cargar EfficientNet-B0 preentrenado
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 128)

        # Capas adicionales
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout después de la primera capa
        self.fc2 = nn.Linear(128, 2)        # Capa de salida
        self.dropout2 = nn.Dropout(p=0.3)   # Dropout antes de la capa de salida

    def forward(self, x):
        x = self.efficientnet(x)  # Pasar por EfficientNet
        x = self.dropout1(x)      # Aplicar Dropout
        x = self.fc2(x)           # Capa intermedia
        x = self.dropout2(x)      # Aplicar Dropout antes de la salida
        return x
