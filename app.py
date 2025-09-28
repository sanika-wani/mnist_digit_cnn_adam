import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import MNISTCNN

# ----------------------------
# Device & Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model.eval()

# ----------------------------
# Page Config & Style
# ----------------------------
st.set_page_config(page_title="MNIST Digit Drawer", layout="centered")
st.markdown("""
<style>
body { background-color: #f0f0f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
.stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
canvas { border: 2px solid #4CAF50; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🖊 MNIST Digit Drawer")
st.markdown("Draw a digit (0-9) below and click **Recognize Digit**!")

# ----------------------------
# Canvas for Drawing
# ----------------------------
canvas_result = st_canvas(
    fill_color="#FFFFFF",       # White background fill
    stroke_width=15,
    stroke_color="#000000",     # Black pen
    background_color="#FFFFFF", # White canvas
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ----------------------------
# Recognize Button
# ----------------------------
if st.button("Recognize Digit"):
    if canvas_result.image_data is not None:
        # Take RGB channels only (ignore alpha)
        img_data = canvas_result.image_data[:, :, :3].astype('uint8')
        img = Image.fromarray(img_data).convert('L')  # grayscale
        img = img.resize((28,28))
        
        # Convert to numpy and invert colors (black pen -> white digit)
        img_array = np.array(img)/255.0
        img_array = 1 - img_array
        
        # Normalize with MNIST mean and std
        img_array = (img_array - 0.1307)/0.3081
        
        # Convert to tensor
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Display prediction
        st.subheader(f"Predicted Digit: {pred}")
        
        # Probability distribution bars
        st.markdown("### Probability Distribution:")
        for i in range(10):
            bar_color = "#4CAF50" if i == pred else "#d3d3d3"
            st.markdown(
                f"<div style='background-color:#e0e0e0; padding:4px; border-radius:5px;'>"
                f"{i}: <span style='color:{bar_color}'>{probs[i]*100:.2f}%</span></div>",
                unsafe_allow_html=True
            ) 