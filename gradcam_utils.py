import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_gradcam(model, input_tensor, original_image):

    target_layers = [model.layer4[-1]]

    # Get predicted class
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(predicted_class)]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    rgb_img = np.array(original_image).astype(np.float32) / 255.0
    rgb_img = cv2.resize(rgb_img, (224, 224))

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization