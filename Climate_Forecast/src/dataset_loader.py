import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def multivariate_sequences(data_path, feature_cols, target_col, seq_length=30):
    df = pd.read_csv(data_path)
    df.columns = [col.strip() for col in df.columns]

    # Extract the metrics from the dataframe
    raw_features = df[feature_cols].values
    raw_target = df[[target_col]].values.reshape(-1, 1)
    # Standardize the features and target
    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(raw_features)
    scaler_target = StandardScaler()
    scaled_target = scaler_target.fit_transform(raw_target)
    
    X_sequences = []
    y_targets = []

    target_idx_in_features = feature_cols.index(target_col)

    for i in range(len(scaled_features) - seq_length):
        history_window = scaled_features[i : i + seq_length]
        target_value = scaled_target[i + seq_length]



        X_sequences.append(history_window)
        y_targets.append(target_value)

    #Convert lists to tensors
    X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_targets), dtype=torch.float32)
    
    return X_tensor, y_tensor, scaler_features, scaler_target, target_idx_in_features
       
