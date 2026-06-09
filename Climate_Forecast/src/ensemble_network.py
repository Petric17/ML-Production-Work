import torch
import torch.nn as nn

from src.network import TimelineAttention
from src.recurrent_network import ClimateGRU

class EnsembleModel(nn.Module):
    def __init__(self, num_features=3, seq_length=30, embed_dim=32):
        super().__init__()

        self.feature_projection = nn.Linear(num_features, embed_dim)
        self.attention_engine = TimelineAttention(embed_dim)
        self.attention_readout = nn.Linear(seq_length * embed_dim, embed_dim)

        self.rnn_engine = nn.GRU(
            input_size=num_features,
            hidden_size=embed_dim, 
            batch_first=True
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, climate_sequences):
        B, T, C = climate_sequences.shape

        proj = self.feature_projection(climate_sequences)
        attn_output = self.attention_engine(proj)
        flattened_attn = attn_output.view(B, -1)
        compiled_attn = self.attention_readout(flattened_attn)

        all_rnn_states, _ = self.rnn_engine(climate_sequences)
        track2_context = all_rnn_states[:, -1, :]
        fused_context = torch.cat([compiled_attn, track2_context], dim=-1)
        final_forecast = self.prediction_head(fused_context)

        return final_forecast
    
