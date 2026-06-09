import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.ensemble_network import EnsembleModel
import matplotlib.pyplot as plt

class BaseExtractor(ABC):
  dd . ; git commit -m "Auto-update" 
    def extract(self) -> pd.DataFrame:
        pass

class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class CSV_Extractor(BaseExtractor):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def extract(self) -> pd.DataFrame:
        print(f"Extracting data from {self.file_path}...")
        df = pd.read_csv(self.file_path)
        df_colums = [col.strip() for col in df.columns]
        return df 

class ClimateTransformer(BaseTransformer):
    def __init__(self, feature_cols: list[str], target_col: str, seq_length: int = 30):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_length = seq_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"Transforming data from {df.columns}...")

        raw_features = df[self.feature_cols].values
        raw_target = df[[self.target_col]].values

        scaled_X = self.scaler_X.fit_transform(raw_features)
        scaled_y = self.scaler_y.fit_transform(raw_target)

        X_seq, y_target = [], []
        for i in range(len(scaled_X) - self.seq_length):
            X_seq.append(scaled_X[i : i + self.seq_length])
            y_target.append(scaled_y[i + self.seq_length])
        
        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_target), dtype=torch.float32)
        return X_tensor, y_tensor
    
class ModelTrainer:
    def __init__(self, epochs: int=50, batch_size: int=128, lr: float=0.005):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    def train(self, X, y):
        print(f"Training model on {X.shape} {y.shape}...")
        print("-----------------------------------------------------")

        split = int(len(X) * 0.85)
        X_train, y_train = X[:split].to(self.device), y[:split].to(self.device)

        num_features = X.shape[2]
        active_model = EnsembleModel(num_features=num_features).to(self.device)
        optimizer = torch.optim.Adam(active_model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()

        for epoch in range(self.epochs):
            active_model.train()
            permutation = torch.randperm(X_train.size()[0])

            for i in range(0, X_train.size()[0], self.batch_size):
                optimizer.zero_grad()
                indices = permutation[i : i + self.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]

                predictions = active_model(batch_x)
                loss = loss_fn(predictions, batch_y)

                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:2d}/{self.epochs} | Total Loss: {loss.item():.5f}")
        
        print("-----------------------------------------------------")
        print(f'Training completed! Neural weights optimized successfully.')
        return active_model

class ModelEvaluator:
    def __init__(self, scaler_y: StandardScaler):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_y = scaler_y

    def evaluate_and_plot(self, model, X, y):
        print(f"Evaluating model on {X.shape} {y.shape}...")
        model.eval()

        split = int(len(X) * 0.85)
        X_val, y_val = X[split:].to(self.device), y[split:]

        with torch.no_grad():
            val_predictions = model(X_val).cpu().numpy()
        
        preds_2d = val_predictions.reshape(-1, 1)
        targets_2d = y_val.numpy().reshape(-1, 1)

        real_predictions = self.scaler_y.inverse_transform(preds_2d).flatten()
        real_targets = self.scaler_y.inverse_transform(targets_2d).flatten()


        plt.figure(figsize=(12, 6))
        plt.plot(real_targets[:60], label="True Temperatures", color="#2C3E50", linewidth=2)
        plt.plot(real_predictions[:60], label="Ensemble Neural Network Forecast", color="#E67E22", linestyle="--", linewidth=2)
        
        plt.title("Ensemble Neural Network Forecast - Vienna Climate", fontsize=14, fontweight='bold')
        plt.xlabel("Days Into the Future", fontsize=11)
        plt.ylabel("Temperature (°C)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        
        print("[System] Rendering active window. Close the plot window to finish the pipeline.")
        plt.show()
    

class Pipeline:
    def __init__(self, extractor: BaseExtractor, transformer: BaseTransformer, evaluator: ModelEvaluator):
        self.extractor = extractor
        self.transformer = transformer
        self.trainer = ModelTrainer()
        self.evaluator = evaluator

    def run(self):
        print("Running pipeline and training model...")
        print("-----------------------------------------------------")

        raw_data = self.extractor.extract()
        X, y = self.transformer.transform(raw_data)
        trained_model = self.trainer.train(X, y)
        self.evaluator.evaluate_and_plot(trained_model, X, y)


if __name__ == "__main__":
    csv_loader = CSV_Extractor(file_path="data/geocoded_data.csv")
    climate_transformer = ClimateTransformer(feature_cols=['Month', 'Day', 'Temp', 'Pressure', 'Humidity'], target_col='Temp')
    chart_generator = ModelEvaluator(scaler_y=climate_transformer.scaler_y)
    pipeline = Pipeline(extractor=csv_loader, transformer=climate_transformer, evaluator=chart_generator)
    pipeline.run()
    print("\u2699\ufe0f [System] Pipeline execution block completed successfully.")

