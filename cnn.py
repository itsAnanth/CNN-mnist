import torch
import torch.utils.data.dataloader
import torchvision as tv

train = tv.datasets.mnist.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
)

test = train = tv.datasets.mnist.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    dataset=train,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test,   
    batch_size=64,
    shuffle=True
)
image, label = train_loader.dataset[0]
print(image.shape, label)