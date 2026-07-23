import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

import matplotlib.pyplot as plt
import numpy as np
import os

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
# Test Transform
# -----------------------------
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Test Dataset
# -----------------------------
test_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model.load_state_dict(
    torch.load(
        "models/pneumonia_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model Loaded Successfully!")

# -----------------------------
# Evaluation
# -----------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:,1].cpu().numpy())

# -----------------------------
# Metrics
# -----------------------------
print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=test_data.classes
))

print("\nConfusion Matrix\n")

cm = confusion_matrix(
    all_labels,
    all_preds
)

print(cm)

auc = roc_auc_score(
    all_labels,
    all_probs
)

print(f"\nROC-AUC Score: {auc:.4f}")

accuracy = np.mean(
    np.array(all_preds) == np.array(all_labels)
)

print(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# Save Confusion Matrix
# -----------------------------
os.makedirs("outputs", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_data.classes
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")