import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def load_image(path):
    image = Image.open(path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def visualize_convolution_steps(image, kernel, num_steps=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image.squeeze(0).squeeze(0), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    gen = image
    # Generate intermediate results
    step_size = 1.0 / num_steps
    for i in range(1, num_steps):
        # Scale the kernel by the current step
        current_kernel = kernel * (i * step_size)
        
        # Apply convolution
        conv_result = F.conv2d(gen, current_kernel, padding=1)
        
        # Normalize output for visualization
        conv_result = conv_result.squeeze(0).squeeze(0).detach().numpy()
        conv_result = (conv_result - conv_result.min()) / (conv_result.max() - conv_result.min())
        gen = torch.tensor(conv_result).unsqueeze(0).unsqueeze(0)
        # Plot
        axes[i].imshow(conv_result, cmap='gray')
        axes[i].set_title(f'Step {i}')
        axes[i].axis('off')
    print(image.shape, conv_result.shape)
    plt.tight_layout()
    plt.show()

def main():
    # Load image
    image_path = "image.jpeg"
    image = load_image(image_path)
    
    # Define kernels
    kernels = {
        'random': torch.randn((1, 3, 3), dtype=torch.float32).unsqueeze(1),

        'edge_detection': torch.tensor([[
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]], dtype=torch.float32).unsqueeze(1),
        
        'blur': torch.tensor([[
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ]], dtype=torch.float32).unsqueeze(1),
        
        'sharpen': torch.tensor([[
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]], dtype=torch.float32).unsqueeze(1)
    }
    
    # Visualize convolution process for each kernel
    for kernel_name, kernel in kernels.items():
        print(f"Visualizing {kernel_name} convolution")
        visualize_convolution_steps(image, kernel)

if __name__ == "__main__":
    main()