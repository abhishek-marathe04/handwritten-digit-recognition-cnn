from PIL import Image
import numpy as np
import torch
from app.inference import make_prediction

img = Image.open("canvas_input.png")

img = img.resize((28, 28))


img_np = np.array(img)



img_np.shape == (28, 28)

img_np = img_np / 255.0
img_np = (img_np - 0.1307) / 0.3081




img_tensor = torch.tensor(img_np, dtype=torch.float32)

print(img_tensor)

img_tensor = img_tensor.unsqueeze(0)

output = make_prediction(input_image=img_tensor)

print(output)

