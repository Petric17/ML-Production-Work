import torch
import torch.nn as nn

class PendulumModel(nn.Module):
    """
    A multi-layer neural network blueprint designed to output 
    a pendulum's angle given a specific time step.
    """
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1, 64)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(64, 1)
        
    def forward(self, time_input):
        """
        Defines the logical sequence of how data flows through the tools.
        """
        x = self.input_layer(time_input)
        x = self.activation(x)
        angle_output = self.output_layer(x)
        return angle_output