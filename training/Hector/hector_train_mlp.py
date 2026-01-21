import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Check for GPU (Cuda), Mac Silicon (MPS), or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# minimalist class containing heritating from the nn.Module that gives ML Layers.
# this class contains only the init method and the forward, 
# init method initializes the layers of the MLP and the flatten metehod to flatten images
# forwaard methods defines how the input data goes throu MLP : here it goes through sequential of MLP Layers
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

# Setup Data (sets batch size, normalisation of MNIST data)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

# Initialize (instanciate the model , the criterion loss and the optimiser method adam)
model = MinimalMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train():
    model.train()
    for epoch in range(15): # nb epochs 
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # takes a batch from trainloader
            optimizer.zero_grad() # reset grad retropropagation
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete.")

# Testing
def test():
    model.eval()
    correct = 0
    total = 0
    
    # torch.no_grad() disables gradient calculation to save memory and speed up testing
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Get the index of the highest value
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Results:")
    print(f"Total Images Tested: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    

def save_weights(model, file_name):
    if not os.path.exists('models'):
        os.makedirs('models')
    path = os.path.join("models", file_name)
    torch.save(model.state_dict(), path)
    print(f"Poids du modèle MLP sauvegardés avec succès dans : {path}")    
    


if __name__ == "__main__":
    train()
    test()
    save_weights(model, "mlp_model.pt")