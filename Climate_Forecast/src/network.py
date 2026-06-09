import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimelineAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape

        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(C)
        attention_weights = F.softmax(scores, dim=-1)
        context_vectors = torch.bmm(attention_weights, values)

        return self.out_projection(context_vectors)

class ClimatePredictor(nn.Module):
    def __init__(self, num_features=3, seq_length=30, embed_dim=32, hidden_dim=128):
        super().__init__()

        self.seq_length = seq_length
        self.embed_dim = embed_dim
        
        self.feature_projection = nn.Linear(num_features, embed_dim)
        self.attention_engine = TimelineAttention(embed_dim)

        self.readout_layer = nn.Linear(seq_length * embed_dim, embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )
    
    def forward(self, climate_sequences):
        B, T, C = climate_sequences.shape
        
        projected_features = self.feature_projection(climate_sequences)
        attention_context = self.attention_engine(projected_features) 
        
        flattened_timeline = attention_context.view(B, -1) 
        compiled_context = self.readout_layer(flattened_timeline) 
        predicted_temperature = self.feed_forward(compiled_context)

        return predicted_temperature
