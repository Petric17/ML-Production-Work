import torch
import torch.nn as nn

class ClimateGRU(nn.Module):
    def __init__(self, num_features=3, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        #The recurrent layer
        self.rnn_engine = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.feed_forward = nn.Linear(hidden_dim, 1)
    
    def forward(self, climate_sequences):
        B, T, C = climate_sequences.shape

        all_hidden_states, final_hidden_state = self.rnn_engine(climate_sequences)
        last_day_context = all_hidden_states[:, -1, :]
        base_prediction = self.feed_forward(last_day_context)
        yesterday_temps = climate_sequences[:, -1, 2].unsqueeze(-1)
        final_forecast = base_prediction + yesterday_temps

        return final_forecast
    
    