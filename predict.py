import torch
import os
from PIL import Image
from torchvision import transforms

# Cargar el modelo
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Procesar y predecir la clase y las probabilidades de una imagen
def process_image(model, image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).squeeze()
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities.tolist()

# Procesar carpeta completa o imagen individual
def evaluate_images(model, path, is_folder):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ajustar según el tamaño de entrada del modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if is_folder:
        results = []
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            if os.path.isfile(img_path):
                predicted_class, probabilities = process_image(model, img_path, transform)
                results.append((img_file, predicted_class, probabilities))
                
                if len(results) >= 10:
                    break  # Mostrar solo las primeras 10 imágenes
        
        for img_file, predicted_class, probabilities in results:
            print(f"Imagen: {img_file}, Clase Predicha: {'Real' if predicted_class == 1 else 'Editada'}, "
                  f"Probabilidades -> Real: {probabilities[1]:.4f}, Editada: {probabilities[0]:.4f}")
    else:
        predicted_class, probabilities = process_image(model, path, transform)
        print(f"Imagen: {path}, Clase Predicha: {'Real' if predicted_class == 1 else 'Editada'}, "
              f"Probabilidades -> Real: {probabilities[1]:.4f}, Editada: {probabilities[0]:.4f}")

# Main
def main():
    model = load_model("/Users/luisgc/Desktop/FakeSceneClassification/model.pt")

    option = input("¿Deseas evaluar la carpeta de test (1) o brindar una imagen (2)?: ")
    if option == '1':
        folder_path = input("/Users/luisgc/Desktop/FakeSceneClassification/cidaut-ai-fake-scene-classification-2024/Test")
        evaluate_images(model, folder_path, is_folder=True)
    elif option == '2':
        image_path = input("Ingrese el path de la imagen: ")
        evaluate_images(model, image_path, is_folder=False)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()
