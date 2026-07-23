import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Class Names
# -----------------------------
class_names = ["NORMAL", "PNEUMONIA"]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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
# Image Path
# -----------------------------
image_path = input("Enter image path: ")

# -----------------------------
# Load Image
# -----------------------------
image = Image.open(image_path).convert("RGB")

plt.imshow(image)
plt.title("Input X-ray")
plt.axis("off")
plt.show()

# -----------------------------
# Prediction
# -----------------------------
tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]

    pred = probs.argmax().item()

print("\nPrediction :", class_names[pred])
print("Confidence :", f"{probs[pred].item()*100:.2f}%")

print("\nProbabilities")

for i, cls in enumerate(class_names):
    print(f"{cls}: {probs[i].item()*100:.2f}%")