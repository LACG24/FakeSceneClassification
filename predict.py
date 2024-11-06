import torch
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from model import CustomEfficientNet
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Definir un dataset personalizado para imágenes
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def predict_image(image_path, model, device, transform):
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Añadir una dimensión para el batch

    # Realizar la predicción
    with torch.no_grad():
        output = model(image)
        probability = torch.softmax(output, dim=1).cpu().numpy()[0]

    # Mostrar la imagen y las probabilidades
    plt.imshow(image.cpu().squeeze().numpy().transpose((1, 2, 0)))
    real_prob, fake_prob = probability[1], probability[0]
    plt.title(f"Real: {real_prob:.2f}, Editada: {fake_prob:.2f}")
    plt.axis('off')
    plt.show()

def predict_images(image_dir_or_path, model_path, device):
    # Transformaciones
    val_test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cargar el modelo
    model = load_model(model_path, device)
    model.eval()

    # Detectar si es una imagen individual o un directorio
    if os.path.isfile(image_dir_or_path):
        # Si es una sola imagen
        predict_image(image_dir_or_path, model, device, val_test_transform)
    elif os.path.isdir(image_dir_or_path):
        # Si es un directorio de imágenes
        test_dataset = CustomImageDataset(image_dir=image_dir_or_path, transform=val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # Obtener y mostrar las predicciones para cada lote
        with torch.no_grad():
            count = 1  # Número de lotes que deseas procesar
            for batch_idx, (images, filenames) in enumerate(test_loader):
                if batch_idx >= count:
                    break  # Detener
                images = images.to(device)
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
            

# Configuración del dispositivo y ruta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir_or_path = input("Enter the path to the image or directory of images: ")
model_path = '/Users/luisgc/Desktop/FakeSceneClassification/modelw.pt'

# Llamada a la función de predicción
predict_images(image_dir_or_path, model_path, device)
