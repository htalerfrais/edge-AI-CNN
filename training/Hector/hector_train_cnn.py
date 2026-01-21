#%%
import torch.nn as nn
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import os

torch.random.manual_seed(0)

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


# instanciate model CNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# Load MNIST dataset
# transform is going to be used when we load the dataset to get it in the right format and normalize it
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std well known (dont have to compute them here because famous dataset)
])

# we load famous datasets thanks to torchvision.dataset, using transform predefined
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# we instanciate dataloaders, with the batch size we want
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

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
