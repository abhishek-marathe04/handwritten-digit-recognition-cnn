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
        height=200,
        width=200,
        stroke_width=10,
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

    img_np = np.array(pil_img)

    # Anything above this is considered "digit", This gives you a mask of where the digit exists.
    threshold = 20
    binary = img_np > threshold

    # We now find the smallest box that contains all True pixels. This gives you all (row, col) locations of the digit.
    coords = np.column_stack(np.where(binary))

    # This defines: [top, left] → [bottom, right]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit tightly, Now, No extra black space, Just digit
    cropped = img_np[y_min:y_max+1, x_min:x_max+1]

    # Make it square. If you resize directly → distortion. Now, Aspect ratio preserved
    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    square_pil = Image.fromarray(square)
    final_img = square_pil.resize((28, 28))

    # # Resize to MNIST size
    st.image(final_img, caption="Centered 28x28 input to model")

    # Convert to NumPy
    img_np = np.array(final_img)

    # Normalize (MNIST)
    img_np = img_np / 255.0
    img_np = (img_np - 0.1307) / 0.3081

    # Tensor: [1, 28, 28]
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

    # Inference
    output, confidence = make_prediction(input_image=img_tensor)

    # ------------------------
    # Display result
    # ------------------------
    prediction_placeholder.markdown(
        f"### ✅ Predicted Digit: **{output}**"
    )
    

elif predict_clicked:
    st.warning("Please draw a digit before clicking Predict.")
