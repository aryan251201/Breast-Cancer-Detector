import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import BreastCancerCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = BreastCancerCNN().to(device)

# Loss functions
criterion_level = nn.CrossEntropyLoss()
criterion_type = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Simulated level labels (1–4)
        level_labels = torch.randint(0, 4, (labels.size(0),)).to(device)

        optimizer.zero_grad()

        level_out, type_out = model(images)

        loss_level = criterion_level(level_out, level_labels)
        loss_type = criterion_type(type_out, labels)

        loss = loss_level + loss_type
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "breast_cancer_model.pth")
print("Model saved successfully.")
