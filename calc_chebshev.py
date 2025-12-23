import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class MultiLayerPiecewiseCheb:
    def __init__(self, layer_configs):
        """
        layer_configs: list of dicts [{"num_pieces": int, "deg": int}, ...]
        Example: [{"num_pieces": 50, "deg": 3}, {"num_pieces": 25, "deg": 2}]
        """
        self.layer_configs = layer_configs
        self.models = []
    
    def fit(self, x_train, y_train):
        """Train multi-layer piecewise Chebyshev network with residual learning."""
        x_current = x_train.copy()
        y_current = y_train.copy()
        
        for layer_idx, config in enumerate(self.layer_configs):
            num_pieces = config["num_pieces"]
            deg = config["deg"]
            print(f"Training layer {layer_idx+1}/{len(self.layer_configs)}: {num_pieces} pieces, deg={deg}")
            
            # Fit this layer
            layer_models = piecewise_cheb_fit(x_current, y_current, deg, num_pieces)
            
            # Safety check
            if not layer_models:
                print(f"WARNING: Layer {layer_idx+1} produced no models - skipping")
                continue
            
            self.models.append(layer_models)
            
            # Residual learning: next layer learns what this one missed
            y_layer_pred = piecewise_cheb_predict(layer_models, x_current)
            residuals = y_current - y_layer_pred
            y_current = residuals  # Update target for next layer
        
        return self
    
    def predict(self, x_data):
        """Forward pass: accumulate all layer outputs."""
        if not self.models:
            return np.zeros_like(x_data)
        
        y_pred = np.zeros_like(x_data)
        for layer_models in self.models:
            if layer_models:
                layer_output = piecewise_cheb_predict(layer_models, x_data)
                y_pred += layer_output  # Residual accumulation
        
        return y_pred


def piecewise_cheb_fit(x_train, y_train, deg_per_piece, num_pieces):
    """Perform piecewise Chebyshev fit using equal-width segments in x-domain."""
    x_min, x_max = x_train.min(), x_train.max()
    edges = np.linspace(x_min, x_max, num_pieces + 1)
    models = []

    for i in range(num_pieces):
        start_edge, end_edge = edges[i], edges[i + 1]
        mask = (x_train >= start_edge) & (x_train < end_edge)
        x_piece = x_train[mask]
        y_piece = y_train[mask]

        if len(x_piece) < deg_per_piece + 1:
            continue

        # Scale to [-1, 1] domain
        x_scaled = 2 * (x_piece - start_edge) / (end_edge - start_edge) - 1
        coeffs = chebfit(x_scaled, y_piece, deg_per_piece)
        models.append((start_edge, end_edge, coeffs))

    return models


def piecewise_cheb_predict(models, x_data):
    """Evaluate Chebyshev models piecewise on new data."""
    y_pred = np.zeros_like(x_data)
    for (start_edge, end_edge, coeffs) in models:
        mask = (x_data >= start_edge) & (x_data < end_edge)
        x_piece = x_data[mask]
        x_scaled = 2 * (x_piece - start_edge) / (end_edge - start_edge) - 1
        y_pred[mask] = chebval(x_scaled, coeffs)
    return y_pred


def calculate_and_evaluate_multilayer(x_data, y_data, layer_configs, test_ratio=0.2):
    """Train/test split and multi-layer Chebyshev evaluation."""
    # Sort data initially
    sort_idx = np.argsort(x_data)
    x_data, y_data = x_data[sort_idx], y_data[sort_idx]

    # Random train/test split
    indices = np.random.permutation(len(x_data))
    split_point = int(len(x_data) * (1 - test_ratio))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    x_train, x_test = x_data[train_idx], x_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]

    # Train multi-layer model
    model = MultiLayerPiecewiseCheb(layer_configs)
    model.fit(x_train, y_train)

    # Predict on test data
    y_test_pred = model.predict(x_test)

    # Compute MSE
    least_squares_error = np.sum((y_test - y_test_pred) ** 2) / len(y_test)

    return least_squares_error, x_test, y_test, y_test_pred, model


def plot_sample_results(x_test, y_test, y_test_pred, sample_indices, filename):
    """Plot the random sample points for verification."""
    plt.style.use('fast')
    plt.rcParams['agg.path.chunksize'] = 10000
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    ax1.scatter(x_test[sample_indices], y_test[sample_indices], c='blue', s=60, label='True', zorder=5)
    ax1.scatter(x_test[sample_indices], y_test_pred[sample_indices], c='red', s=60, marker='^', label='Predicted', zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Multi-Layer Piecewise Chebyshev: True vs Predicted')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as: {filename}")


def run_chebyshev_evaluation(x_data, y_data, layer_configs, test_ratio, num_samples, filename):
    """Run full multi-layer Chebyshev evaluation."""
    least_squares_error, x_test, y_test, y_test_pred, model = calculate_and_evaluate_multilayer(
        x_data, y_data, layer_configs, test_ratio
    )

    print(f"Multi-layer Chebyshev MSE: {least_squares_error:.6f}")
    print(f"Layers trained: {len(model.models)}")

    # Sample random indices for plotting
    sample_indices = np.random.choice(len(y_test), size=num_samples, replace=False)
    plot_sample_results(x_test, y_test, y_test_pred, sample_indices, filename)

def nd_cheb_projection(X, num_moments=5):
    """Project N-D to 1D using Chebyshev polynomial moments."""
    N_samples, N_dims = X.shape
    
    # Normalize each dimension to [-1,1]
    X_norm = np.zeros_like(X)
    for d in range(N_dims):
        x_min, x_max = X[:,d].min(), X[:,d].max()
        X_norm[:,d] = 2*(X[:,d] - x_min)/(x_max - x_min) - 1
    
    # Compute Chebyshev moments per dimension, weighted sum
    projection = np.zeros(N_samples)
    for d in range(N_dims):
        for k in range(num_moments):
            # k-th Chebyshev moment of dimension d
            coeffs = np.polynomial.chebyshev.chebfit(X_norm[:,d], X_norm[:,d], k+1)
            T_k = np.polynomial.chebyshev.chebval(X_norm[:,d], coeffs)
            projection += 1.0/(N_dims * num_moments) * T_k
    
    return projection  # Now feed to your 1D Cheb model!



# Example usage
if __name__ == "__main__":
    # One dimensional test data
    num_data_points = 1_000_000
    x = np.linspace(0, 10, num_data_points)
    y = np.sin(x)  # Pure sine for testing

    # 3-layer configuration using dicts: coarse → medium → fine
    layer_configs = [
        {"num_pieces": 10, "deg": 3},   # Layer 1: global fit
        {"num_pieces": 10, "deg": 5},   # Layer 2: local corrections
        {"num_pieces": 50, "deg": 2}    # Layer 3: fine residuals
    ]

    run_chebyshev_evaluation(
        x_data=x,
        y_data=y,
        layer_configs=layer_configs,
        test_ratio=0.4,
        num_samples=100,
        filename='multilayer_cheb_plot.png'
    )

    # N-D test data
    N_dims = 5
    X_nd = np.random.rand(num_data_points, N_dims) * 10
    y_nd = (np.sin(X_nd[:,0]) + np.cos(X_nd[:,1]) + 
            0.5*X_nd[:,2] + 0.1*np.sin(X_nd[:,3]*X_nd[:,4])).flatten()
    from sklearn.random_projection import SparseRandomProjection

    lsh = SparseRandomProjection(n_components=1, random_state=42)
    X_proj = lsh.fit_transform(X_nd).flatten()
    multi_layer_configs = [
        {"num_pieces": 100, "deg": 20},   # Layer 1: global fit
        {"num_pieces": 10, "deg": 5},   # Layer 2: local corrections
        {"num_pieces": 5, "deg": 5}    # Layer 3: fine residuals
    ]
    run_chebyshev_evaluation(
        x_data=X_proj,
        y_data=y_nd,
        layer_configs=multi_layer_configs,
        test_ratio=0.1,
        num_samples=100,
        filename='nd_multilayer_cheb_plot.png'
    )
