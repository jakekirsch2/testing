import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        self.net = nn.Sequential(*layers, nn.Linear(prev_dim, output_dim))
    
    def forward(self, x):
        return self.net(x)

def fit_model(X, y, test_ratio=0.2, hidden_layers=(128, 64, 32), 
              epochs=200, lr=1e-3, batch_size=1024, filename='model'):
    """Arbitrary input/output dims - SIMPLIFIED"""
    # Auto-detect shapes
    input_dim = X.shape[1]
    if y.ndim == 1:
        output_dim = 1
        y = y.reshape(-1, 1)
    else:
        output_dim = y.shape[1]
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    
    # Model + training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(input_dim, hidden_layers, output_dim).to(device)
    
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss {losses[-1]:.4f}')
    
    # Test
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_pred = model(X_test_t).cpu().numpy()
    
    mse = np.mean((y_test - y_pred) ** 2)
    print(f'Test MSE: {mse:.6f}')
    
    plot_results(X_test, y_test, y_pred, losses, filename)
    return model

def plot_results(X_test, y_test, y_pred, losses, filename, n_points=100):
    idx = np.random.choice(len(y_test), min(n_points, len(y_test)), replace=False)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    x1d = pca.fit_transform(X_test[idx]).flatten()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x1d, y_test[idx], c='blue', s=30, label='True', alpha=0.7)
    plt.scatter(x1d, y_pred[idx], c='red', s=30, label='Pred', alpha=0.7, marker='^')
    plt.xlabel('PCA1'); plt.ylabel('y'); plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses); plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved: {filename}.png')

# TEST: Continuous, vector output length 1
if __name__ == '__main__':
    np.random.seed(42)
    n = 10000
    d = 1_000_000
    X = np.random.randn(n, d)
    y = (np.sin(X[:, 0]) + 0.5 * X[:, 1]**2 + np.cos(X[:, 2])).reshape(-1, 1)
    
    model = fit_model(X, y, hidden_layers=(10, 10, 10, 10), epochs=200, filename='simple_test')
    
