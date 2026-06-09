import torch
import torch.nn as nn

class AsteroidPhysicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3, 128)
        self.activation = nn.SiLU()
        self.hidden_layer1 = nn.Linear(128, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, features):
        x1 = self.activation(self.input_layer(features))
        x2 = self.activation(self.hidden_layer1(x1))
        x3 = self.activation(self.hidden_layer2(x2))
        output = self.output_layer(x3)
        return output
    
