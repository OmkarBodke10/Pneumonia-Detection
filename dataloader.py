import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset path
DATA_DIR = "data/chest_xray"

# Training transforms
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # Convert grayscale to RGB
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test transforms
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
train_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transforms
)

val_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transforms
)

test_data = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=val_transforms
)

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Classes:", train_data.classes)
print("Class Mapping:", train_data.class_to_idx)

print(f"Training Images   : {len(train_data)}")
print(f"Validation Images : {len(val_data)}")
print(f"Testing Images    : {len(test_data)}")

# Check one batch
images, labels = next(iter(train_loader))

print("\nBatch Shape :", images.shape)
print("Labels Shape:", labels.shape)