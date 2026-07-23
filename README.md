\# 🩺 Pneumonia Detection using Deep Learning



An end-to-end Deep Learning web application built with \*\*PyTorch\*\* and \*\*Streamlit\*\* to detect \*\*Pneumonia\*\* from Chest X-ray images. The app provides real-time confidence scores, visual explanations using \*\*Grad-CAM\*\*, and downloadable PDF diagnostic reports.



🚀 \*\*Live App:\*\* \[Pneumonia Detection App](https://pneumonia-detection-mctdtimjdi4qvjizwlornj.streamlit.app/)



\---



\## 📌 Features



\* 🔬 \*\*AI-Powered Diagnosis:\*\* Classifies Chest X-ray scans as \*\*NORMAL\*\* or \*\*PNEUMONIA\*\*.

\* 🔥 \*\*Grad-CAM Visualizations:\*\* Highlights key region-of-interest heatmaps so users can see where the model is looking.

\* 📊 \*\*Confidence \& Probabilities:\*\* Displays class probability distributions for transparency.

\* 📄 \*\*Automated PDF Reports:\*\* Allows users to download a summary report of the diagnosis.

\* ☁️ \*\*Cloud Deployed:\*\* Hosted on Streamlit Community Cloud.



\---



\## 📊 Model Performance



The classifier is based on a fine-tuned \*\*ResNet18\*\* architecture trained on Chest X-ray datasets.



| Metric | Score |

| :--- | :--- |

| \*\*Model Architecture\*\* | ResNet18 (PyTorch) |

| \*\*Test Accuracy\*\* | \*\*88.62%\*\* |

| \*\*ROC-AUC Score\*\* | \*\*95.42%\*\* |



\---



\## 🛠️ Tech Stack



\* \*\*Deep Learning Framework:\*\* PyTorch, Torchvision

\* \*\*Model Explainability:\*\* `pytorch-grad-cam`

\* \*\*Web Framework:\*\* Streamlit

\* \*\*PDF Generation:\*\* `fpdf2`

\* \*\*Image Processing:\*\* OpenCV, Pillow

\* \*\*Data Processing \& Visualization:\*\* NumPy, Pandas, Matplotlib, Scikit-Learn



\---



\## 📁 Repository Structure



```text

Pneumonia-Detection/

├── .streamlit/

│   └── config.toml         # Streamlit UI configuration

├── models/

│   └── pneumonia\_resnet18.pth # Trained PyTorch weights

├── app.py                  # Main Streamlit web application

├── gradcam\_utils.py        # Helper module for Grad-CAM generation

├── packages.txt            # System dependencies (libgl1 for OpenCV)

├── requirements.txt        # Python package dependencies

└── README.md               # Project documentation

