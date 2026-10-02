import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM Model aligned with the MIT thesis.
    CNN extracts spatial/feature relationships from the PCA-reduced inputs.
    LSTM models temporal dependencies.
    """
    def __init__(self, input_dim, seq_len, cnn_filters=32, lstm_hidden=64, output_dim=1):
        super(CNNLSTM, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 1D CNN over the sequence length. 
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        lstm_input_size = cnn_filters
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(lstm_hidden, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1) # (batch_size, input_dim, seq_len)
        
        c = self.conv1(x)
        c = self.relu(c)
        c = self.pool(c)
        
        c = c.permute(0, 2, 1) # (batch_size, seq_len/2, cnn_filters)
        
        lstm_out, _ = self.lstm(c)
        last_out = lstm_out[:, -1, :] # Get last time step output
        
        out = self.fc(last_out)
        return out

def create_sequences(X_data, y_data, seq_length):
    xs = []
    ys = []
    for i in range(len(X_data) - seq_length):
        xs.append(X_data[i:(i + seq_length)])
        ys.append(y_data[i + seq_length])
    return np.array(xs), np.array(ys)

def train_and_predict_alpha(asset_returns, benchmark_returns, betas, pca_features_dict, seq_length=20, epochs=50, batch_size=32, lr=0.001):
    """
    Trains CNN-LSTM models on PCA features and predicts expected returns.
    Then calculates Expected Alpha = E[R_i] - Beta_i * E[R_m]
    """
    print("\n--- Training Hybrid CNN-LSTM Models on PCA Features ---")
    expected_alphas = {}
    
    # Calculate historical expected return of benchmark as proxy for E[R_m]
    expected_market_return = benchmark_returns.mean()
    print(f"Historical Expected Market Return (Daily): {expected_market_return:.6f}")
    
    for asset in asset_returns.columns:
        print(f"Training model for {asset}...")
        
        X_asset = pca_features_dict[asset]
        y_asset = asset_returns[asset].values.reshape(-1, 1)
        
        # We assume X_asset is already scaled/PCA'd, we only scale Y for training
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_asset)
        
        X_seq, y_seq = create_sequences(X_asset, y_scaled, seq_length)
        
        # Train/test split (80/20)
        split = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = X_asset.shape[1]
        model = CNNLSTM(input_dim=input_dim, seq_len=seq_length)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Compute Stochastic Gradient Vector L2 Norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Stochastic Grad Norm: {grad_norm:.6f}")
                
        # Generate prediction for the next day
        last_seq = X_asset[-seq_length:]
        last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            pred_scaled = model(last_seq_tensor).numpy()
            
        pred_actual = scaler_y.inverse_transform(pred_scaled)
        expected_return = pred_actual[0][0]
        
        # Alpha Calculation (MIT Thesis Eq 1.1 modified for Expected Alpha)
        # alpha_i = R_ei - beta_i * R_m
        beta = betas[asset]
        expected_alpha = expected_return - (beta * expected_market_return)
        
        expected_alphas[asset] = expected_alpha
        
    print("\n--- Predicted Expected Alphas for Portfolio Optimization ---")
    alpha_df = pd.Series(expected_alphas, name='Expected Alpha')
    print(alpha_df)
    
    return alpha_df
