import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch

from inference import make_prediction

# ------------------------
# Page config
# ------------------------
st.set_page_config(
    page_title="Draw a Digit",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Draw a Digit")
st.caption("Handwritten Digit Recognition using a CNN trained on MNIST")

st.markdown(
    "Draw a digit **(0–9)** in the box below and click **Predict**."
)

st.divider()

# ------------------------
# Layout
# ------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🖌️ Canvas")

    canvas_result = st_canvas(
        background_color="black",
        stroke_color="white",
        height=220,
        width=220,
        stroke_width=12,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_clicked = st.button("🔍 Predict", use_container_width=True)

with col2:
    st.subheader("📊 Prediction")

    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()

st.divider()

# ------------------------
# Preprocessing & Inference
# ------------------------
if canvas_result.image_data is not None and predict_clicked:

    # Canvas image (RGBA)
    img = canvas_result.image_data

    # Convert to PIL
    pil_img = Image.fromarray(img.astype(np.uint8), mode="RGBA")

    # Convert to grayscale
    pil_img = pil_img.convert("L")

    # Resize to MNIST size
    pil_img_28 = pil_img.resize((28, 28))

    # Convert to NumPy
    img_np = np.array(pil_img_28)

    # Normalize (MNIST)
    img_np = img_np / 255.0
    img_np = (img_np - 0.1307) / 0.3081

    # Tensor: [1, 28, 28]
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

    # Inference
    output = make_prediction(input_image=img_tensor)

    # ------------------------
    # Display result
    # ------------------------
    prediction_placeholder.markdown(
        f"### ✅ Predicted Digit: **{output}**"
    )

elif predict_clicked:
    st.warning("Please draw a digit before clicking Predict.")
