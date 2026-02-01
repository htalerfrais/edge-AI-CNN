import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


class MinimalMLP(nn.Module):
    def __init__(self):
        super(MinimalMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def evaluate_local_dataset(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prétraitement MNIST (28x28, grayscale)
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

    model = MinimalMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    
    print("\n--- Analyse des images en temps réel ---")
    print(f"{'Image':<10} | {'Réel':<6} | {'Prédit':<6} | {'Statut'}")
    print("-" * 40)

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
            
            status = "✅" if is_correct else "❌"
            print(f"n°{i+1:<8} | {label_val:<6} | {pred_val:<6} | {status}")

    accuracy = 100 * correct / total
    print("-" * 40)
    print(f"Précision finale : {accuracy:.2f}% ({correct}/{total})")
    
    return all_labels, all_preds

def plot_confusion_matrix(all_labels, all_preds, save_path=None):
    """Génère et sauvegarde la matrice de confusion."""
    if not all_labels:
        return
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("Matrice de Confusion - MLP (dataset de test)")
    plt.xlabel("Prédictions")
    plt.ylabel("Labels réels")
    plt.tight_layout()
    if save_path is None:
        save_path = MODELS_DIR / "confusion_matrix_mlp.png"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée : {save_path}")


def test_single_image(model_path, image_path):
    """Inférence sur une image : affiche les scores de la dernière couche et la prédiction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    raw_image = Image.open(image_path)
    input_tensor = transform(raw_image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    model = MinimalMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        scores = outputs[0]

    _, predicted = torch.max(outputs, 1)
    pred_val = predicted.item()

    print("-" * 30)
    print("Scores de la dernière couche :")
    for i in range(len(scores)):
        print(f"  Classe {i} = {scores[i].item():.4f}")
    print("-" * 30)
    print(f"Image : {image_path}")
    print(f"Chiffre prédit : {pred_val}")
    return pred_val


if __name__ == "__main__":
    MODEL_FILE = MODELS_DIR / "mlp_model.pt"
    DATA_PATH = BASE_DIR.parent / "data" / "mnist_digit_test"
    IMAGE_PATH = BASE_DIR.parent / "data" / "mnist_digit_test" / "2" / "digit_2_2.bmp"  # exemple

    print("=== Test sur le dataset de test (MLP) ===\n")
    labels, preds = evaluate_local_dataset(str(MODEL_FILE), str(DATA_PATH))
    if labels:
        plot_confusion_matrix(labels, preds)

    print("\n=== Inférence sur une image (scores dernière couche) ===\n")
    test_single_image(str(MODEL_FILE), str(IMAGE_PATH))