import torch
import torch.utils.data.dataloader
import torchvision as tv
import os 

trainDir = tv.datasets.mnist.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
)

testDir = tv.datasets.mnist.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    dataset=trainDir,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=testDir,   
    batch_size=64,
    shuffle=True
)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN()
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to model file
model_path = os.path.join(current_dir, 'model.pth')
trainMode = not (os.path.isfile(model_path))
print(model_path)

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch: {i}, Loss: {loss.item()}')
            

    torch.save(model.state_dict(), model_path)
    print("saved model parameters")

if (trainMode):
    print("Model parameters not found, switching to model training")
    train(model, train_loader)
    print("Model trained")
else:
    print("Model parameters found, switching to model evaluation")
    model.load_state_dict(torch.load(model_path))
    print("loaded model parameters")

model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
# torch.load()