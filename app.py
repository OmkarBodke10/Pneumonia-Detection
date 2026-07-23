import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fpdf import FPDF

# Import Grad-CAM visualizer helper
from gradcam_utils import generate_gradcam

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.title("🩺 Pneumonia Detection")

    st.markdown("## 🤖 Model")
    st.write("ResNet18 (PyTorch)")

    st.markdown("## 📊 Performance")
    st.success("Accuracy : 88.62%")
    st.success("ROC-AUC : 95.42%")

    st.markdown("---")

    st.info(
        "⚠️ This application is for educational purposes only "
        "and should not be used for medical diagnosis."
    )

# ==========================================
# Title & Header
# ==========================================
st.title("🩺 Pneumonia Detection using Deep Learning")

st.markdown("""
Upload a **Chest X-ray image** and the trained **ResNet18**
model will predict whether the lungs are **NORMAL**
or affected by **PNEUMONIA**.
""")

col1, col2 = st.columns(2)

with col1:
    st.metric("Test Accuracy", "88.62%")

with col2:
    st.metric("ROC-AUC", "95.42%")

st.markdown("---")

# ==========================================
# Setup Device & Cached Model Loader
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_pneumonia_model():
    """Load model architecture and weights safely using weights_only=True."""
    model = models.resnet18(weights=None)
    
    # Custom classifier head matching training structure
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )

    model.load_state_dict(
        torch.load(
            "models/pneumonia_resnet18.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.to(device)
    model.eval()
    return model

model = load_pneumonia_model()

# ==========================================
# Class Names & Image Transforms
# ==========================================
class_names = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# PDF Report Generator Helper
# ==========================================
def generate_pdf_report(prediction, confidence, normal_prob, pneumonia_prob):
    """Generates a formatted PDF report in bytes."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 12, "PNEUMONIA DETECTION REPORT", ln=True, align="C")
    pdf.ln(5)
    
    # Horizontal Rule
    pdf.set_line_width(0.5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)
    
    # Diagnosis Section
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Diagnosis Summary", ln=True)
    
    pdf.set_font("Arial", "", 11)
    pdf.cell(55, 7, "Prediction:", ln=False)
    
    pdf.set_font("Arial", "B", 11)
    if prediction == "PNEUMONIA":
        pdf.set_text_color(180, 40, 40)
    else:
        pdf.set_text_color(40, 140, 40)
    pdf.cell(0, 7, prediction, ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)
    pdf.cell(55, 7, "Confidence:", ln=False)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, f"{confidence:.2f}%", ln=True)
    pdf.ln(4)
    
    # Probabilities Section
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Class Probabilities", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(55, 7, "Normal Probability:", ln=False)
    pdf.cell(0, 7, f"{normal_prob:.2f}%", ln=True)
    pdf.cell(55, 7, "Pneumonia Probability:", ln=False)
    pdf.cell(0, 7, f"{pneumonia_prob:.2f}%", ln=True)
    pdf.ln(4)
    
    # Model Performance Section
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Model Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(55, 7, "Architecture:", ln=False)
    pdf.cell(0, 7, "ResNet18 (PyTorch)", ln=True)
    pdf.cell(55, 7, "Test Accuracy:", ln=False)
    pdf.cell(0, 7, "88.62%", ln=True)
    pdf.cell(55, 7, "ROC-AUC:", ln=False)
    pdf.cell(0, 7, "95.42%", ln=True)
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This report is generated automatically for educational purposes "
        "only and should not be used as a substitute for professional medical diagnosis or clinical judgment."
    )
    
    return bytes(pdf.output())

# ==========================================
# Upload Image & Inference Section
# ==========================================
uploaded_file = st.file_uploader(
    "📂 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    img_col, result_col = st.columns([1, 1])

    with img_col:
        st.image(
            image,
            caption="Uploaded Chest X-ray",
            width="stretch"
        )

    # Preprocess tensor
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    confidence = probabilities[predicted_class].item() * 100

    # Display Results
    with result_col:
        st.subheader("Prediction Result")

        if predicted_class == 0:
            st.success(f"✅ {class_names[predicted_class]}")
        else:
            st.error(f"⚠️ {class_names[predicted_class]}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown("---")

    # Probability Distributions
    st.subheader("Prediction Probabilities")

    # NORMAL Probability
    st.write(f"🟢 NORMAL : {probabilities[0].item() * 100:.2f}%")
    st.progress(float(probabilities[0]))

    # PNEUMONIA Probability
    st.write(f"🔴 PNEUMONIA : {probabilities[1].item() * 100:.2f}%")
    st.progress(float(probabilities[1]))

    # ==========================================
    # Grad-CAM Visualization
    # ==========================================
    st.markdown("---")
    st.subheader("🔥 Grad-CAM Visualization")

    heatmap = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        original_image=image
    )

    st.image(
        heatmap,
        caption="Grad-CAM Heatmap",
        width="stretch"
    )

    # ==========================================
    # Download PDF Prediction Report
    # ==========================================
    st.markdown("---")
    st.subheader("📄 Download Prediction Report")

    pdf_bytes = generate_pdf_report(
        prediction=class_names[predicted_class],
        confidence=confidence,
        normal_prob=probabilities[0].item() * 100,
        pneumonia_prob=probabilities[1].item() * 100
    )

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="Pneumonia_Report.pdf",
        mime="application/pdf"
    )

# ==========================================
# Footer
# ==========================================
st.markdown("---")

st.warning(
    "⚠️ This application is intended for educational purposes only "
    "and should not be used as a substitute for professional medical diagnosis."
)