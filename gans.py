import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),  # Input: CIFAR-10 images (3 channels, 32x32 pixels)
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Binary output for classification
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.model(x)

# Prepare data (CIFAR-10 dataset as an example)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
discriminator = Discriminator()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # For demonstration, assume labels are binary (you may need to adjust this for your task)
        labels = torch.ones(inputs.size(0))  # Example: all labels are set to 1 for positive class
        labels = labels.unsqueeze(1)  # Reshape labels to match the output size [64, 1]

        # Move inputs and labels to CPU (since you're not using CUDA)
        inputs, labels = inputs.to('cpu'), labels.to('cpu')

        # Forward pass
        outputs = discriminator(inputs)
        loss = criterion(outputs, labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:  # Print every 100 steps
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training complete.")
torch.save(discriminator.state_dict(), 'discriminator.pth')

