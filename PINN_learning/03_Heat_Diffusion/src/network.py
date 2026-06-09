import torch
import torch.nn as nn

class HeatDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 64)
        self.activation = nn.Tanh()
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x_and_t_input):
        h = self.activation(self.input_layer(x_and_t_input))
        h = self.activation(self.hidden_layer(h))
        temperature_output = self.output_layer(h)
        return temperature_output
    