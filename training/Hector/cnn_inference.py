import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from hector_train_cnn import CNN




def evaluate_local_dataset(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prétraitement pour format MNIST (28x28, niveaux de gris) [cite: 168, 202]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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


if __name__ == "__main__":
    MODEL_FILE = "models/cnn_model.pt"
    DATA_PATH = "../../data/mnist_digit/" 
    
    labels, preds = evaluate_local_dataset(MODEL_FILE, DATA_PATH)