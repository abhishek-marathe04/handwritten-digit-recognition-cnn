import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image
import numpy as np
import torch

from inference import make_prediction

SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("Drawable Canvas")
st.markdown("""
Draw on the canvas, get the image data back into Python !
* Doubleclick to remove the selected object when not in drawing mode
""")
st.sidebar.header("Configuration")


predict_clicked = st.button("Predict")

# Create a canvas component
image_data = st_canvas(
    background_color="black", stroke_color="white", height=200, drawing_mode='freedraw', key="canvas", width=200, stroke_width=10
)

if image_data.image_data is not None:
    img = image_data.image_data
    st.write(img.shape)
    st.image(img)

    # Convert NumPy → PIL Image
    pil_img = Image.fromarray(img.astype(np.uint8), mode="RGBA")


    # 🔑 convert to grayscale
    pil_img = pil_img.convert("L")
    # # Save
    # file_path = os.path.join(SAVE_DIR, "canvas_input.png")
    # pil_img.save(file_path)

    # st.success(f"Image saved to {file_path}")

    # gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
    # gray_pil = Image.fromarray(gray, mode="L")
    # gray_pil.save("saved_images/canvas_input_gray.png")

    img = pil_img.resize((28, 28))

    st.image(img, caption="28x28 input")

    img_np = np.array(img)

    img_np.shape == (28, 28)

    img_np = img_np / 255.0
    img_np = (img_np - 0.1307) / 0.3081




    img_tensor = torch.tensor(img_np, dtype=torch.float32)

    img_tensor = img_tensor.unsqueeze(0)

   # 🔑 inference ONLY here
    if predict_clicked:
        output = make_prediction(input_image=img_tensor)
        st.write("**Output:**", output)