import torch
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from model import CustomEfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import CustomEfficientNet

# Definir un dataset personalizado
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]

def load_model(model_path, device):
    model = CustomEfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # Cargar solo los pesos
    return model

# Transformaciones
val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Crear el DataLoader para el conjunto de imágenes
image_dir = '/Users/luisgc/Desktop/FakeSceneClassification/cidaut-ai-fake-scene-classification-2024/Test'
test_dataset = CustomImageDataset(image_dir=image_dir, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Obtener 10 imágenes del conjunto de datos de test
images, filenames = next(iter(test_loader))
images = images.to(device)

# Cargar el modelo
model = load_model('/Users/luisgc/Desktop/FakeSceneClassification/modelw.pt', device)
model.eval()

# Obtener las predicciones del modelo
with torch.no_grad():
    outputs = model(images)

# Calcular las probabilidades
probabilities = torch.softmax(outputs, dim=1)

# Mostrar las imágenes y las probabilidades
fig, axes = plt.subplots(5, 2, figsize=(10, 15))
for i, (image, probability, filename) in enumerate(zip(images, probabilities, filenames)):
    image = image.cpu().numpy().transpose((1, 2, 0))
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = np.clip(image, 0, 1)
    ax = axes[i // 2, i % 2]
    ax.imshow(image)
    ax.set_title(f"Image Name: {filename}")
    ax.axis('off')

    # Mostrar las probabilidades
    real_prob = probability[1].item()
    fake_prob = probability[0].item()
    ax.text(0.5, -0.1, f"Real: {real_prob:.2f}, Editada: {fake_prob:.2f}", ha='center', transform=ax.transAxes)

plt.tight_layout()
plt.show()