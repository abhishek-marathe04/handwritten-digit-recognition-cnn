
import torch
from torch import nn

class MNISTModelV0(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  padding=1,
                  kernel_size=3,
                  stride=1
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
          out_channels=hidden_units,
          padding=1,
          kernel_size=3,
          stride=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  padding=1,
                  kernel_size=3,
                  stride=1
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
          out_channels=hidden_units,
          padding=1,
          kernel_size=3,
          stride=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classification_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units * 7 * 7,
                  out_features=output_shape)
    )

  def forward(self, x):
    x = self.conv_layer_1(x)
    # print(f"Shape of x after conv layer 1 {x.shape}")
    x = self.conv_layer_2(x)
    # print(f"Shape of x after conv layer 2 {x.shape}")
    x = self.classification_layer(x)
    # print(f"Shape of x after classification_layer  {x.shape}")
    return x