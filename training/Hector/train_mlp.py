import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import os

# --- CONFIGURATION & MODELE ---
torch.random.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
PERSO_DATA_PATH = "../../data/mnist_digit/" # Vérifie bien ce chemin
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

# --- CALCUL DES STATISTIQUES GLOBALES ---
def get_global_stats():
    base_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    mnist_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_tf)
    perso_raw = torchvision.datasets.ImageFolder(root=PERSO_DATA_PATH, transform=base_tf)
    
    train_len = int(0.8 * len(perso_raw))
    perso_train_raw, _ = random_split(perso_raw, [train_len, len(perso_raw) - train_len])
    
    loader = DataLoader(ConcatDataset([mnist_raw, perso_train_raw]), batch_size=1024)
    
    mean, std, nb_samples = 0., 0., 0
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    return mean / nb_samples, std / nb_samples



def train(epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")

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
    perso_full = torchvision.datasets.ImageFolder(root=PERSO_DATA_PATH, transform=final_tf)

    # Split Perso : 80% train, 20% test
    train_size = int(0.8 * len(perso_full))
    perso_train, perso_test = random_split(perso_full, [train_size, len(perso_full) - train_size],
                                        generator=torch.Generator().manual_seed(0))

    # Concaténation avec oversampling du perso (x50)
    train_dataset = ConcatDataset([mnist_train] + [perso_train] * 50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Test uniquement sur les données perso jamais vues
    test_loader = DataLoader(perso_test, batch_size=BATCH_SIZE, shuffle=False)

    # --- ENTRAINEMENT & TEST ---
    model = MinimalMLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train(10)
    test()
    
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), "models/mlp_model.pt")