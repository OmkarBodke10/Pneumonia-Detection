import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model.load_state_dict(
    torch.load("models/pneumonia_resnet18.pth", map_location=device)
)

model.to(device)
model.eval()

# Last convolutional layer
target_layers = [model.layer4[-1]]

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = image.resize((224,224))

rgb_img = np.array(image).astype(np.float32) / 255

input_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output).item()

class_names = ["NORMAL", "PNEUMONIA"]

print("Prediction:", class_names[pred])

# Grad-CAM
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [ClassifierOutputTarget(pred)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM")
plt.axis("off")

plt.tight_layout()
plt.show()