import os
from model import MNISTModelV0
import torch

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level and then into 'models' folder
model_path = os.path.join(current_dir, "..", "models", "handwritten_digit_recogniser.pth")


class_names = ['0 - zero',
 '1 - one',
 '2 - two',
 '3 - three',
 '4 - four',
 '5 - five',
 '6 - six',
 '7 - seven',
 '8 - eight',
 '9 - nine']

def make_prediction(input_image):
    print(input_image.shape)
    mnist_model = MNISTModelV0(input_shape=1, hidden_units=10, output_shape=10)

    state_dict = torch.load(model_path)
    mnist_model.load_state_dict(state_dict)

    mnist_model.eval()
    #Forward pass
    with torch.no_grad():
        pred_logit = mnist_model(input_image.unsqueeze(0))

        # Get predictions probablity (logits -> Predictions probablity)
        pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

        pred_label = class_names[torch.argmax(pred_prob)]

    return pred_label


