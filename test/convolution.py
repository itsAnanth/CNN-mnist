from PIL import Image
import torch
from torchvision import transforms
import os 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



def load_image(path) -> torch.Tensor:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    image = Image.open(abs_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    return transform(image).unsqueeze(0)


"""
    kernel should be of shape (out, in, height, width)
    where out is the number of output channels
    in is the number of input channels
    height and width are the kernel dimensions
    
    out our case the out channel is 3 for RGB
    in is 1, but we're processing conv2d in groups of 3, so each channel is processed independently
"""
kernels = {
    'blur': torch.tensor([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
    
    'edge': torch.tensor([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
}

def convolute(image, kernel):
    """Preserve RGB channels during convolution"""
    # groups=1 since we're using proper channel dimensions now
    feature_map = F.conv2d(image, kernel, stride=1, padding=1, groups=3)
    return feature_map

def visualize(image, kernel, epochs=5):
    fig, axes = plt.subplots(1, epochs + 1, figsize=(15, 3))
    feature_map = image
    
    # Original image display
    axes[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i in range(1, epochs + 1):
        feature_map = convolute(feature_map, kernel)
        print(feature_map.shape)
        
        # Normalize each channel independently
        for c in range(feature_map.shape[1]):  # Iterate over channels
            channel = feature_map[0, c]
            channel = channel - channel.min()
            channel = channel / (channel.max() + 1e-8)
            feature_map[0, c] = channel

        # RGB display
        img_display = feature_map.squeeze(0).permute(1, 2, 0).detach().numpy()
        axes[i].imshow(img_display)
        axes[i].set_title(f'Step {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test code
img = load_image('../image.jpeg')
print(img.shape)
visualize(img, kernels['edge'], epochs=5)

