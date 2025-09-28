# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model import MNISTCNN  # Your CNN model class

# ----------------------------
# Device & Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")
st.title("🎨 Draw a digit and see the prediction!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=15,
    stroke_color="#FFFFFF",  # White pen
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_image(img):
    # Convert RGBA to grayscale
    img = img.convert("L")
    # Resize to 28x28
    img = img.resize((28, 28))
    # Invert colors (MNIST is black on white)
    img = np.array(img)
    img = 255 - img
    # Normalize like MNIST
    img = img / 255.0
    img = (img - 0.1307) / 0.3081
    # Convert to tensor
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

# ----------------------------
# Predict Button
# ----------------------------
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    if st.button("Predict Digit"):
        input_tensor = preprocess_image(img)
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
        st.success(f"Predicted Digit: {pred}")
