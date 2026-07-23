import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -----------------------------
# Dataset Path
# -----------------------------
DATA_DIR = "data/chest_xray"

# -----------------------------
# Image Transformations
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Load Dataset
# -----------------------------
train_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transforms
)

val_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transforms
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False
)

# -----------------------------
# Class Weights
# -----------------------------
labels = [label for _, label in train_data.samples]

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
).to(device)

print("Class Weights:", class_weights)

# -----------------------------
# Load Pretrained ResNet18
# -----------------------------
weights = models.ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(weights=weights)

# Freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,2)
)

model = model.to(device)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-4
)

# -----------------------------
# Training Function
# -----------------------------
def train_one_epoch():

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()*images.size(0)

        _,predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss/total
    epoch_acc = correct/total

    return epoch_loss,epoch_acc

# -----------------------------
# Validation Function
# -----------------------------
def validate():

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs,labels)

            running_loss += loss.item()*images.size(0)

            _,predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss/total
    epoch_acc = correct/total

    return epoch_loss,epoch_acc

# -----------------------------
# Train Model
# -----------------------------
best_acc = 0

os.makedirs("models",exist_ok=True)

epochs = 10

for epoch in range(epochs):

    train_loss,train_acc = train_one_epoch()

    val_loss,val_acc = validate()

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" | Train Loss: {train_loss:.4f}"
        f" | Train Acc: {train_acc:.4f}"
        f" | Val Loss: {val_loss:.4f}"
        f" | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            "models/pneumonia_resnet18.pth"
        )

        print("Best Model Saved!")

print("\nTraining Completed!")