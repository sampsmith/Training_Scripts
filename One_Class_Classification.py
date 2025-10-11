import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# -----------------------------
# 1. Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "./nail_dataset"  # Folder with all positive images
batch_size = 16
num_epochs = 20
lr = 1e-4
embedding_dim = 512  # size of feature embedding

# -----------------------------
# 2. Dataset & Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# 3. Backbone Model
# -----------------------------
backbone = models.resnet18(pretrained=True)
backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)  # Output embedding
backbone = backbone.to(device)

# -----------------------------
# 4. Compute hypersphere center c (optional: initialize after first batch)
# -----------------------------
# We can start with c as zero vector
c = torch.zeros(embedding_dim, device=device)

# -----------------------------
# 5. Deep SVDD Loss
# -----------------------------
def deep_svdd_loss(embeddings, center):
    dist = torch.sum((embeddings - center) ** 2, dim=1)
    return torch.mean(dist)

# -----------------------------
# 6. Optimizer
# -----------------------------
optimizer = optim.Adam(backbone.parameters(), lr=lr)

# -----------------------------
# 7. Training Loop
# -----------------------------
for epoch in range(num_epochs):
    backbone.train()
    epoch_loss = 0
    for images, _ in dataloader:  # labels not used
        images = images.to(device)
        embeddings = backbone(images)
        
        # Update center c after first batch (optional for initialization)
        if epoch == 0:
            c = torch.mean(embeddings.detach(), dim=0)

        loss = deep_svdd_loss(embeddings, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
    
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# -----------------------------
# 8. Save model
# -----------------------------
torch.save(backbone.state_dict(), "deep_svdd_nail.pth")
print("Model saved.")
