import streamlit as st
import cv2
import torch
import numpy as np
import time

st.title("Depth Estimation with MiDaS")

# Load a MiDas model for depth estimation
model=st.selectbox("Select the model type:",["MiDaS v3-Large(highest accuracy, slowest inference speed)","MiDaS v3-Hybrid(medium accuracy, medium inference speed)","MiDaS v2.1-Small(lowest accuracy, highest inference speed)"])

if model=="MiDaS v3-Large(highest accuracy, slowest inference speed)":
    model_type = "DPT_Large" 

elif model=="MiDaS v3-Hybrid(medium accuracy, medium inference speed)":
    model_type = "DPT_Hybrid" 

elif model=="MiDaS v2.1-Small(lowest accuracy, highest inference speed)":
    model_type = "MiDaS_small" 

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Allow users to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image using OpenCV
    start=time.time()
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(image).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    end = time.time()  # Stop measuring time

    inference_time = end - start
    fps = 1.0 / inference_time  # Calculate frames per second

    depth_map = prediction.cpu().numpy()
    
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    col1, col2 = st.columns(2)

    col1.image(image, channels="RGB", use_column_width=True, caption="Uploaded Image")

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    col2.image(depth_map, channels="BGR", use_column_width=True, caption="Depth Map")
   
    # Display inference speed
    st.write(f"Inference Speed: {fps:.2f} FPS")
