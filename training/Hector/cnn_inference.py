import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from hector_train_cnn import CNN
from PIL import Image


def evaluate_local_dataset(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prétraitement pour format MNIST (28x28, niveaux de gris) [cite: 168, 202]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        custom_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
        print(f"Dataset local chargé : {len(custom_dataset)} images trouvées.")
    except Exception as e:
        print(f"Erreur lors du chargement du dossier : {e}")
        return [], []
    
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(custom_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            label_val = labels.item()
            pred_val = predicted.item()
            
            all_labels.append(label_val)
            all_preds.append(pred_val)
            
            total += 1
            is_correct = (pred_val == label_val)
            if is_correct:
                correct += 1
            
            # Affichage au fur et à mesure dans le terminal
            status = "✅" if is_correct else "❌"
            print(f"n°{i+1:<8} | {label_val:<6} | {pred_val:<6} | {status}")

    accuracy = 100 * correct / total
    print("-" * 40)
    print(f"Précision finale : {accuracy:.2f}% ({correct}/{total})")
    
    return all_labels, all_preds



def test_single_image(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    raw_image = Image.open(image_path)
    input_tensor = transform(raw_image) # Donne un tenseur [1, 28, 28]
    
    input_tensor = input_tensor.unsqueeze(0).to(device) # [1, 1, 28, 28]

    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_tensor)
        scores = outputs[0] 
        _, predicted = torch.max(outputs, 1)
    
    print("-" * 25)
    print("Dernière couche : \n")
    for i in range(len(scores)):
        print(f"Score {i} = {scores[i].item():.4f}")
    print("-" * 25)
    print(f"Fichier testé : {image_path}")
    print(f"Chiffre détecté : {predicted.item()}")
    
    return predicted.item()



if __name__ == "__main__":
    MODEL_FILE = "models/cnn_model.pt"
    DATA_PATH = "../../data/mnist_digit/" 
    IMAGE_PATH = "../../data/mnist_digit/2/digit_2_6.bmp"
    
    # labels, preds = evaluate_local_dataset(MODEL_FILE, DATA_PATH)
    
    test_single_image(MODEL_FILE, IMAGE_PATH)