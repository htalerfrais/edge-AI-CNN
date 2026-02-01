#%%
import torch.nn as nn
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import os
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader


torch.random.manual_seed(0)

# --- CONFIGURATION & MODÈLE ---
torch.random.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PERSO_TRAIN_PATH = "../data/mnist_digit_train"
PERSO_VAL_PATH = "../data/mnist_digit_val"
PERSO_TEST_PATH = "../data/mnist_digit_test"
BATCH_SIZE = 64
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(7*7*32, self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.flatten(start_dim = 1))
        return out


# ---  CALCUL DES STATISTIQUES GLOBALES ---

def get_global_stats():
    # Transform minimal pour calcul (sur train uniquement)
    base_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    mnist_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_tf)
    perso_train_raw = torchvision.datasets.ImageFolder(root=PERSO_TRAIN_PATH, transform=base_tf)
    
    loader = DataLoader(ConcatDataset([mnist_raw, perso_train_raw]), batch_size=1024)
    
    mean = 0.
    std = 0.
    nb_samples = 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    return mean / nb_samples, std / nb_samples



# Def training loop (with validation each epoch)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return train_losses, train_accs, val_losses, val_accs
        
        
        
# Def testing loop

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Calculate final metrics
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy


def plot_metrics(train_losses, train_accs, val_losses, val_accs, save_path=None):
    """Trace et sauvegarde les courbes loss et accuracy (train / val)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.set_title("Accuracy")
    plt.tight_layout()
    if save_path is None:
        save_path = MODELS_DIR / "metrics_cnn.png"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Courbes sauvegardées : {save_path}")


def save_weights(model, file_name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / file_name
    torch.save(model.state_dict(), str(path))
    print(f"Poids du modèle CNN sauvegardés avec succès dans : {path}")    
    




if __name__ == "__main__":

    # --- DATASETS FINAUX ---
    # transform with normalisation adapted to concatenated dataset

    print("Calcul des statistiques globales...")
    mean_val, std_val = get_global_stats()
    print(f"Stats calculées : Mean={mean_val.item():.4f}, Std={std_val.item():.4f} \n !! à appliquer aussi pour l'inférence C !!")

    final_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean_val.item(),), (std_val.item(),))
    ])

    # Chargement MNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=final_tf)

    # Chargement train / val / test (dossiers créés par utils/split_dataset.py)
    perso_train = torchvision.datasets.ImageFolder(root=PERSO_TRAIN_PATH, transform=final_tf)
    perso_val = torchvision.datasets.ImageFolder(root=PERSO_VAL_PATH, transform=final_tf)
    perso_test = torchvision.datasets.ImageFolder(root=PERSO_TEST_PATH, transform=final_tf)

    train_dataset = ConcatDataset([mnist_train] + [perso_train] * 50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(perso_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(perso_test, batch_size=BATCH_SIZE, shuffle=False)




    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Starting training...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10
    )
    plot_metrics(train_losses, train_accs, val_losses, val_accs)
    print("\nStarting testing...")
    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    save_weights(model, "cnn_model.pt")
