#%%
import torch.nn as nn
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import os
from torch.utils.data import ConcatDataset, DataLoader, random_split


torch.random.manual_seed(0)

# --- CONFIGURATION & MODÈLE ---
torch.random.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PERSO_DATA_PATH = "../../data/mnist_digit/" # À adapter
BATCH_SIZE = 64

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
    # Transform minimal pour calcul
    base_tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    mnist_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=base_tf)
    perso_raw = torchvision.datasets.ImageFolder(root=PERSO_DATA_PATH, transform=base_tf)
    
    # On split pour ne calculer que sur la partie "train" du perso
    train_len = int(0.8 * len(perso_raw))
    perso_train_raw, _ = random_split(perso_raw, [train_len, len(perso_raw) - train_len])
    
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

# Chargement & Split Perso
perso_full = torchvision.datasets.ImageFolder(root=PERSO_DATA_PATH, transform=final_tf)
train_size = int(0.8 * len(perso_full))
perso_train, perso_test = random_split(perso_full, [train_size, len(perso_full) - train_size],
                                       generator=torch.Generator().manual_seed(0))

# Concaténation + Oversampling (Perso x50 pour équilibrer face aux 60k MNIST)
train_dataset = ConcatDataset([mnist_train] + [perso_train] * 50)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(perso_test, batch_size=BATCH_SIZE, shuffle=False)




# instanciate model CNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Definition of loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




# Def training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        # the model sees all the batches, calculates a loss that aggregates for each batch process
        running_loss = 0.0  
        correct = 0
        total = 0
        
        # batch_idx n'exista pas au prealable, il est crée par le enumerate
        for batch_idx, (data, target) in enumerate(train_loader):
            # for batch (here data is the data of a whole batch, target too) :
            data, target = data.to(device), target.to(device)
            
            # Zero gradients before doing grad update for this new batch 
            optimizer.zero_grad()
            
            # Forward pass and loss on the whole batch
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass for the whole batch
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() # loss s'accumulant sur touts les batchs
            _, predicted = outputs.max(1) # prediction made on the batch
            total += target.size(0)
            correct += predicted.eq(target).sum().item() # how much predictions were right within the data prediction in this current batch 
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        # Optional: Update learning rate
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed, '
              f'Average Loss: {running_loss/len(train_loader):.4f}, '
              f'Accuracy: {100.*correct/total:.2f}%')
        
        
        
# Def testing loop

def test_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for testing
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


def save_weights(model, file_name):
    if not os.path.exists('models'):
        os.makedirs('models')
    path = os.path.join("models", file_name)
    torch.save(model.state_dict(), path)
    print(f"Poids du modèle CNN sauvegardés avec succès dans : {path}")    
    


# call training and testing

if __name__ == "__main__" :
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)
    print("\nStarting testing...")
    test_loss, test_accuracy = test_model(model, test_loader, criterion)
    
    save_weights(model, "cnn_model.pt")
