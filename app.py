import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection using Deep Learning")
st.write("Upload a chest X-ray image to predict whether it is NORMAL or PNEUMONIA.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model.to(device)
model.eval()

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
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose a Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)[0]

        predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class].item() * 100

    st.markdown("---")

    if predicted_class == 0:
        st.success(f"✅ Prediction: {class_names[predicted_class]}")
    else:
        st.error(f"⚠️ Prediction: {class_names[predicted_class]}")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    st.write(f"**NORMAL:** {probabilities[0].item()*100:.2f}%")
    st.progress(float(probabilities[0].item()))

    st.write(f"**PNEUMONIA:** {probabilities[1].item()*100:.2f}%")
    st.progress(float(probabilities[1].item()))