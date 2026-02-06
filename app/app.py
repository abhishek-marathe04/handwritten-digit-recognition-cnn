import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch

# ------------------------
# Model loading (example)
# ------------------------
# Assumes your model and make_prediction logic already exists
# Replace this with your actual model loading code
from inference import make_prediction


# ------------------------
# Helper: show feature maps
# ------------------------
def show_feature_maps(feature_maps, title, max_maps=8):
    st.subheader(title)

    num_maps = min(feature_maps.shape[0], max_maps)
    cols = st.columns(4)

    for i in range(num_maps):
        fm = feature_maps[i].cpu().numpy()
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-5)

        with cols[i % 4]:
            st.image(fm, caption=f"Filter {i}", clamp=True)


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
# Inference + Explanation
# ------------------------
if canvas_result.image_data is not None and predict_clicked:

    # ------------------------
    # Preprocessing
    # ------------------------
    img = canvas_result.image_data
    pil_img = Image.fromarray(img.astype(np.uint8), mode="RGBA").convert("L")

    img_np = np.array(pil_img)

    # Threshold to find digit
    binary = img_np > 20
    coords = np.column_stack(np.where(binary))

    if len(coords) == 0:
        st.warning("Please draw a digit before predicting.")
        st.stop()

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_np[y_min:y_max+1, x_min:x_max+1]

    # Make square
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    final_img = Image.fromarray(square).resize((28, 28))

    st.image(final_img, caption="🧠 Centered 28×28 input seen by the model", width=120)

    # Normalize (MNIST)
    img_np = np.array(final_img) / 255.0
    img_np = (img_np - 0.1307) / 0.3081

    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

    # ------------------------
    # Prediction
    # ------------------------
    pred_digit, confidence, feature_maps_conv1, feature_maps_conv2 = make_prediction(
        input_image=img_tensor
    )

    prediction_placeholder.markdown(
        f"### ✅ Predicted Digit: **{pred_digit}**"
    )

    confidence_placeholder.metric(
        label="Confidence",
        value=f"{confidence:.2f} %"
    )

    st.info(
        "Curious how the CNN arrived at this prediction? "
        "Scroll down to explore how the model processes your digit 👇"
    )

    # ------------------------
    # Explanation Section
    # ------------------------
    st.divider()
    st.subheader("📚 Model Explanation")
    st.caption("A peek inside how the CNN learns and recognizes handwritten digits.")

    with st.expander("🧠 Step-by-step: How the CNN understands your digit"):

        st.markdown(
            """
            **CNN learning pipeline**
            1️⃣ Pixels → edges  
            2️⃣ Edges → shapes  
            3️⃣ Shapes → digit prediction  
            """
        )

        st.caption("🔹 Conv1 learns simple patterns like edges and strokes.")
        show_feature_maps(
            feature_maps_conv1,
            "Conv1 – Edge & Stroke Detectors",
            max_maps=8
        )

        st.caption("🔹 Conv2 combines edges into higher-level shapes and parts.")
        show_feature_maps(
            feature_maps_conv2,
            "Conv2 – Shape & Part Detectors",
            max_maps=12
        )

elif predict_clicked:
    st.warning("Please draw a digit before clicking Predict.")
