import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
batch_size = 10
image_size = 64
channels = 3
latent_dim = 128
num_epochs = 2
learning_rate = 0.0002
beta1 = 0.5
timesteps = 10
num_samples = 50  # Number of samples to save at the end
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a folder for saving the output images
output_folder = "SD_results"
os.makedirs(output_folder, exist_ok=True)

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, latent_dim, 4, 1, 0)
        )

    def forward(self, x):
        return self.encoder(x)

# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

# Define the diffusion process
class Diffuser(nn.Module):
    def __init__(self, latent_dim, timesteps):
        super(Diffuser, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

# Define the pipeline
class Pipeline(nn.Module):
    def __init__(self, encoder, decoder, diffuser):
        super(Pipeline, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffuser = diffuser

    def forward(self, x, t):
        latent = self.encoder(x)
        diffused = self.diffuser(latent, t)
        reconstruction = self.decoder(diffused)
        return reconstruction

# Load dataset
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root='archive/img_align_celeba', transform=transform)
subset_size = int(0.8 * len(dataset))  # Adjust percentage as needed
subset_dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

# Use subset_dataset for training DataLoader
dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
diffuser = Diffuser(latent_dim, timesteps).to(device)
pipeline = Pipeline(encoder, decoder, diffuser).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(pipeline.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Initialize list to store losses
losses = []

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.size(0),), device=device).long()

        # Forward pass
        outputs = pipeline(images, t)
        loss = criterion(outputs, images)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        # Store loss
        losses.append(loss.item())

# Save multiple sample images after training
with torch.no_grad():
    saved_samples = 0
    while saved_samples < num_samples:
        for images, _ in dataloader:
            if saved_samples >= num_samples:
                break
            images = images.to(device)
            t = torch.randint(0, timesteps, (images.size(0),), device=device).long()
            final_outputs = pipeline(images, t)
            
            for output in final_outputs:
                if saved_samples >= num_samples:
                    break
                sample = output.cpu().numpy().transpose(1, 2, 0)
                sample = (sample + 1) / 2  # denormalize
                plt.imsave(os.path.join(output_folder, f'final_sample_image_{saved_samples + 1}.png'), sample)
                saved_samples += 1

# Plot the loss graph
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.show()

print("Training completed!")
