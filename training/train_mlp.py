import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import os

# --- CONFIGURATION & MODELE ---
torch.random.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PERSO_TRAIN_PATH = "../data/mnist_digit_train"
PERSO_VAL_PATH = "../data/mnist_digit_val"
PERSO_TEST_PATH = "../data/mnist_digit_test"
BATCH_SIZE = 64

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

# --- CALCUL DES STATISTIQUES GLOBALES (sur train uniquement) ---
def get_global_stats():
    base_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    mnist_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_tf)
    perso_train_raw = torchvision.datasets.ImageFolder(root=PERSO_TRAIN_PATH, transform=base_tf)
    
    loader = DataLoader(ConcatDataset([mnist_raw, perso_train_raw]), batch_size=1024)
    
    mean, std, nb_samples = 0., 0., 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    return mean / nb_samples, std / nb_samples



def train(epochs=10, val_loader=None):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f"\nAccuracy sur ton dataset PERSO : {100 * correct / total:.2f}%")



if __name__ == "__main__":

    print("Calcul des statistiques globales (MNIST + Perso)...")
    mean_val, std_val = get_global_stats()
    print(f"Stats : Mean={mean_val.item():.4f}, Std={std_val.item():.4f}")

    # --- PREPARATION DES DATASETS ---
    final_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean_val.item(),), (std_val.item(),))
    ])

    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=final_tf)
    perso_train = torchvision.datasets.ImageFolder(root=PERSO_TRAIN_PATH, transform=final_tf)
    perso_val = torchvision.datasets.ImageFolder(root=PERSO_VAL_PATH, transform=final_tf)
    perso_test = torchvision.datasets.ImageFolder(root=PERSO_TEST_PATH, transform=final_tf)

    train_dataset = ConcatDataset([mnist_train] + [perso_train] * 50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(perso_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(perso_test, batch_size=BATCH_SIZE, shuffle=False)

    # --- ENTRAINEMENT & TEST ---
    model = MinimalMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train(10, val_loader=val_loader)
    test()
    
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), "models/mlp_model.pt")