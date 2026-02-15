print("hello worild")
"""
Generative Fashion Design Tool
Author: Simran
Description: Starter script for AI-powered fashion design using diffusion models.
"""

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os

# Placeholder generator model (replace with your GAN or diffusion model)
class FashionGenerator(nn.Module):
    def __init__(self):
        super(FashionGenerator, self).__init__()
        self.fc = nn.Linear(100, 256)  # latent vector to features
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 784) # output image (28x28 for demo)

    def forward(self, z):
        x = self.fc(z)
        x = self.relu(x)
        x = self.out(x)
        return x.view(-1, 28, 28)

def generate_design():
    # Random latent vector
    z = torch.randn(1, 100)
    model = FashionGenerator()
    with torch.no_grad():
        design = model(z)

    # Convert to image
    img = design.squeeze().numpy()
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = Image.fromarray(img.astype("uint8"))
    img.save("fashion_design.png")
    print("✅ Fashion design generated and saved as fashion_design.png")

if __name__ == "__main__":
    print("🎨 Generative Fashion Design Tool")
    generate_design()
