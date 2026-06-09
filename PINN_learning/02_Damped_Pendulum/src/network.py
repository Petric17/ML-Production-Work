import torch
import torch.nn as nn

class DampedPendulumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1, 128)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, time_input):
        x = self.input_layer(time_input)
        x = self.activation(x)
        angle_output = self.output_layer(x)
        return angle_output