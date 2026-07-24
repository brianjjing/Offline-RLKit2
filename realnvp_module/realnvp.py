import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import argparse
import os
import sys
import pickle
try:  # __main__/plotting/config-only deps; RealNVP.load_model + score_samples don't need them
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import seaborn as sns
    import yaml
    import gym
except ModuleNotFoundError:
    pass
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:  # training/plotting-only deps; RealNVP.load_model + score_samples don't need them
    from common.buffer import ReplayBuffer
    from common.util import load_dataset_with_validation_split, create_synthetic_data
    from helpers.plotter import (
        plot_likelihood_distributions,
        plot_val_test_id_distribution,
        plot_ood_metrics,
        plot_roc_curves,
        plot_likelihood_histograms,
        plot_tsne
    )
except ModuleNotFoundError:
    pass  # only used by load_rl_data_for_kde and __main__ (train in GORMPO, not here)


class MLP(nn.Module):
    """Multi-layer perceptron for coupling layer transformations."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CouplingLayer(nn.Module):
    """RealNVP coupling layer with affine transformation."""

    def __init__(self, input_dim: int, hidden_dims: List[int], mask: torch.Tensor):
        super().__init__()

        self.register_buffer('mask', mask)
        masked_dim = int(mask.sum().item())

        # Scale and translation networks
        self.scale_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)
        self.translate_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through coupling layer.

        Args:
            x: Input tensor
            reverse: If True, compute inverse transformation

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.mask.device)
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        # Get scale and translation from masked components
        x_masked_input = x_masked[:, self.mask.bool()]
        scale = self.scale_net(x_masked_input)
        translate = self.translate_net(x_masked_input)

        if not reverse:
            # Forward transformation: y = x * exp(s) + t
            log_scale = torch.tanh(scale)  # Stabilize scale
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = x_unmasked_vals * torch.exp(log_scale) + translate

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = log_scale.sum(dim=1)

        else:
            # Inverse transformation: x = (y - t) * exp(-s)
            log_scale = torch.tanh(scale)
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = (x_unmasked_vals - translate) * torch.exp(-log_scale)

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = -log_scale.sum(dim=1)

        return y, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow model for density estimation."""

    def __init__(
        self,
        input_dim: int =2,
        num_layers: int = 6,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.device = device

        # Create alternating masks
        self.masks = []
        for i in range(num_layers):
            mask = torch.zeros(input_dim)
            if i % 2 == 0:
                mask[::2] = 1  # Even indices
            else:
                mask[1::2] = 1  # Odd indices
            self.masks.append(mask)

        # Create coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dims, mask.to(device))
            for mask in self.masks
        ])

        # Prior distribution parameters (standard normal)
        self.register_buffer('prior_mean', torch.zeros(input_dim))
        self.register_buffer('prior_std', torch.ones(input_dim))

        # Threshold for anomaly detection
        self.threshold = None

    def _apply(self, fn):
        """Override _apply to update self.device when model is moved."""
        super()._apply(fn)
        # Update self.device to match the actual device of parameters
        if len(list(self.parameters())) > 0:
            self.device = next(self.parameters()).device
        return self

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            reverse: If True, sample from the model

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        # Use the actual device of model parameters instead of self.device
        model_device = next(self.parameters()).device
        log_det_total = torch.zeros(x.shape[0], device=model_device)

        if not reverse:
            # Forward: data -> latent
            z = x
            for layer in self.coupling_layers:
                z, log_det = layer(z, reverse=False)
                log_det_total += log_det
        else:
            # Reverse: latent -> data
            z = x
            for layer in reversed(self.coupling_layers):
                z, log_det = layer(z, reverse=True)
                log_det_total += log_det

        return z, log_det_total

    def _log_prob_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability and return as tensor (for training)."""
        z, log_det = self.forward(x, reverse=False)

        # Log probability under prior (standard normal)
        log_prior = -0.5 * (
            z.pow(2).sum(dim=1) +
            self.input_dim * np.log(2 * np.pi)
        )

        return log_prior + log_det

    def score_samples(self, x: torch.Tensor, device='cuda') -> np.ndarray:
        """Compute log probability of data points."""
        # Convert to tensor if numpy
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        # Move to model's device
        model_device = next(self.parameters()).device
        x = x.to(model_device)

        log_prob = self._log_prob_tensor(x)
        return log_prob.detach().cpu().numpy()

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the model."""
        with torch.no_grad():
            # Sample from prior
            model_device = next(self.parameters()).device
            z = torch.randn(num_samples, self.input_dim, device=model_device)

            # Transform to data space
            x, _ = self.forward(z, reverse=True)

        return x

    def fit(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Train the RealNVP model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset for threshold selection
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//2
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('-inf')
        patience_counter = 0

        self.train()
        val_data = (val_data + 1e-3)
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0

            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                # noise = torch.rand_like(batch_data)*0.01   # uniform [0,1)
                batch_data = (batch_data + 1e-3)
                optimizer.zero_grad()

                # Compute negative log likelihood
                log_prob = self._log_prob_tensor(batch_data)
                loss = -log_prob.mean()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation
            self.eval()
            with torch.no_grad():
                # noise = torch.rand_like(val_data)*0.01  # uniform [0,1)
                
                val_log_prob = self.score_samples(val_data.to(self.device))
                val_loss = -val_log_prob.mean().item()
                # print("VALIDATION", val_log_prob.max(), np.exp(val_log_prob.max().item()))
            self.train()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if verbose and epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Early stopping
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

        # Set threshold based on validation data
        self.set_threshold(val_data)

        return history

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01
    ):
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(val_data.to(self.device))

        # Set threshold as percentile of validation log probabilities
        self.threshold = np.quantile(log_probs, anomaly_fraction)

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data

        Returns:
            Boolean tensor indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(x.to(self.device))

        return log_probs < self.threshold
    
    def predict(self, X):
        """
        Predict anomalies based on threshold

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores.cpu() >= self.threshold, 1, -1)
    
    def evaluate_anomaly_detection(
    self,
    normal_data: torch.Tensor,
    anomaly_data: torch.Tensor,
    plot: bool = True) -> dict:
        """
        Evaluate anomaly detection performance.
        """
        self.eval()

        # >>> Get the true device of the model <<<
        model_device = next(self.parameters()).device

        # >>> Move data to the same device as the model <<<
        normal_data = normal_data.to(model_device)
        anomaly_data = anomaly_data.to(model_device)

        print("Model device:", model_device)
        print("Normal data device:", normal_data.device)

        with torch.no_grad():
            normal_log_probs = self.score_samples(normal_data).cpu().numpy()
            anomaly_log_probs = self.score_samples(anomaly_data).cpu().numpy()

        # Create labels (0 = normal, 1 = anomaly)
        y_true = np.concatenate([
            np.zeros(len(normal_log_probs)),
            np.ones(len(anomaly_log_probs))
        ])

        # Use negative log prob as anomaly score
        y_scores = np.concatenate([-normal_log_probs, -anomaly_log_probs])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy with current threshold
        if self.threshold is not None:
            predictions = np.concatenate([
                normal_log_probs < self.threshold,
                anomaly_log_probs < self.threshold
            ])
            accuracy = (predictions == y_true).mean()
        else:
            accuracy = None

        results = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'normal_log_prob_mean': normal_log_probs.mean(),
            'normal_log_prob_std': normal_log_probs.std(),
            'anomaly_log_prob_mean': anomaly_log_probs.mean(),
            'anomaly_log_prob_std': anomaly_log_probs.std()
        }

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Anomaly Detection')
            plt.legend()
            plt.grid(True)
            plt.show()

        return results

    def save_model(self, save_path: str, train_data: torch.Tensor = None):
        """
        Save the RealNVP model and metadata.

        Args:
            save_path: Base path for saving (without extension)
            train_data: Training data for computing statistics (optional)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), f"{save_path}_model.pth")

        # Calculate the logprobs in training data if provided
        if train_data is not None:
            self.eval()
            with torch.no_grad():
                # Ensure train_data is a tensor
                if not isinstance(train_data, torch.Tensor):
                    train_data = torch.FloatTensor(train_data)

                # Move to model's device
                model_device = next(self.parameters()).device
                train_data = train_data.to(model_device)

                train_log_probs = self.score_samples(train_data)
        else:
            train_log_probs = None
        # Save metadata (threshold and config)
        metadata = {
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'hidden_dims': self.hidden_dims,
            'device': str(next(self.parameters()).device),
        }

        if train_log_probs is not None:
            metadata["mean"] = float(train_log_probs.mean())
            metadata["std"] = float(train_log_probs.std())

        with open(f"{save_path}_meta_data.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pth")
        print(f"Metadata saved to: {save_path}_meta_data.pkl")

    @classmethod
    def load_model(cls, save_path: str, hidden_dims: List[int] = [256, 256], device=None):
        """
        Load a saved RealNVP model.

        Args:
            save_path: Base path for loading (without extension)
            hidden_dims: Hidden layer dimensions (must match saved model)
            device: Target device to load the model onto. If None, uses the device
                    stored in the model metadata (i.e., the device it was saved on).

        Returns:
            Loaded RealNVP model
        """
        # Load metadata
        with open(f"{save_path}_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Use provided device or fall back to the saved device
        load_device = str(device) if device is not None else metadata['device']

        # Create model with saved configuration
        model = cls(
            input_dim=metadata['input_dim'],
            num_layers=metadata['num_layers'],
            hidden_dims=hidden_dims,
            device=load_device
        )

        # Load model state dict
        model.load_state_dict(torch.load(f"{save_path}_model.pth", map_location=load_device))
        # model.to(cls.device)

        # Restore threshold
        model.threshold = metadata['threshold']

        print(f"Model loaded from: {save_path}_model.pth")
        print(f"Metadata loaded from: {save_path}_meta_data.pkl")
        print(f"Threshold: {model.threshold}")
        model_dict = {'model': model, 'thr': model.threshold,
                      'threshold_candidates': metadata.get('threshold_candidates', {})}
        return model_dict


def create_synthetic_data(n_samples=1000, dim=2, anomaly_type="outlier"):
    """
    Generate synthetic normal and anomalous data in arbitrary dimensions.

    Args:
        n_samples (int): number of normal samples
        dim (int): dimensionality of data
        anomaly_type (str): "outlier" or "uniform"

    Returns:
        (torch.FloatTensor, torch.FloatTensor): normal_data, anomaly_data
    """
    normal_data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.7:
            # Main cluster around 0
            mean = np.zeros(dim)
            cov = np.eye(dim)                      # identity covariance
            sample = np.random.multivariate_normal(mean, cov, 1)
        else:
            # Secondary cluster around 3
            mean = np.ones(dim) * 3
            cov = 0.5 * np.eye(dim)                # smaller spread
            sample = np.random.multivariate_normal(mean, cov, 1)
        normal_data.append(sample[0])

    normal_data = np.array(normal_data)

    # Anomalous data
    if anomaly_type == "outlier":
        mean = np.ones(dim) * 10
        cov = 2 * np.eye(dim)
        anomaly_data = np.random.multivariate_normal(mean, cov, n_samples // 5)
    elif anomaly_type == "uniform":
        anomaly_data = np.random.uniform(-5, 8, (n_samples // 5, dim))
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return torch.FloatTensor(normal_data), torch.FloatTensor(anomaly_data)


def load_rl_data_for_kde(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL dataset and prepare next_observations + actions for KDE training.

    Args:
        args: Arguments containing data_path, obs_shape, action_dim
        env: Environment object (for Abiomed datasets)
        val_split_ratio: Fraction of data for validation

    Returns:
        Tuple of (train_data, val_data, kde_input_dim)
    """
    # Load dataset using the existing utility function
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio
    )

    train_dataset = dataset_result['train_data']
    val_dataset = dataset_result['val_data']
    test_dataset = dataset_result['test_data']
    buffer_len = dataset_result['buffer_len']

    # print(f"Loaded dataset with {buffer_len} total samples")

    # Handle size mismatch between next_observations and other arrays
    # Some datasets have one fewer next_observation (last state has no next state)
    for dataset_split in [train_dataset, val_dataset, test_dataset]:
        min_len = min(
            len(dataset_split['observations']),
            len(dataset_split['next_observations']),
            len(dataset_split['actions']),
            len(dataset_split['rewards']),
            len(dataset_split['terminals'])
        )
        # Truncate all arrays to the minimum length to ensure consistency
        for key in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            if key in dataset_split:
                dataset_split[key] = dataset_split[key][:min_len]

    # Initialize ReplayBuffer

    offline_buffer = ReplayBuffer(
            buffer_size=len(train_dataset["observations"]),
            obs_shape=args.obs_dim,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
            )
    for i in range(len(train_dataset['observations'])):
        offline_buffer.add(
            obs=train_dataset['observations'][i],
            next_obs=train_dataset['next_observations'][i],
            action=train_dataset['actions'][i],
            reward=train_dataset['rewards'][i],
            terminal=train_dataset['terminals'][i]
        )

    # Extract concatenated next_observations + actions for KDE
    train_next_obs = torch.FloatTensor(train_dataset['next_observations'])
    train_actions = torch.FloatTensor(train_dataset['actions'])

    # Concatenate next_observations and actions for training data
    train_kde_input = torch.cat([train_next_obs, train_actions], dim=1)


    val_next_obs = torch.FloatTensor(val_dataset['next_observations'])
    val_actions = torch.FloatTensor(val_dataset['actions'])


    test_next_obs = torch.FloatTensor(test_dataset['next_observations'])
    test_actions = torch.FloatTensor(test_dataset['actions'])
    # Concatenate next_observations and actions for validation data
    val_kde_input = torch.cat([val_next_obs, val_actions], dim=1)
    test_kde_input = torch.cat([test_next_obs, test_actions], dim=1)

    # Calculate input dimension for KDE model
    kde_input_dim = train_kde_input.shape[1]

    print(f"KDE input shape: {train_kde_input.shape}")
    print(f"KDE input dimension: {kde_input_dim}")
    print(f"Next observations dim: {train_next_obs.shape[1]}, Actions dim: {train_actions.shape[1]}")

    return train_kde_input, val_kde_input, test_kde_input, kde_input_dim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command line arguments."""
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="configs/realnvp/hopper.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
   
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Overrides config file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Overrides config file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output. Overrides config file.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. Overrides config file.')

    # RL dataset specific arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to RL dataset file (pickle or npz)')
    parser.add_argument('--task', type=str, default='hopper-medium-v2',
                        help='Task name (e.g., abiomed, halfcheetah-medium-v2)')
    parser.add_argument('--obs_dim', type=int, nargs='+', default=[11],
                        help='Observation shape (default: [17] for HalfCheetah)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension (default: 6 for HalfCheetah)')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save or load model')
    parser.add_argument('--fig_save_path', type=str, default='figures/',)
    parser.add_argument(
        '--compute_thresholds_only',
        action='store_true',
        help='Skip training: load existing model, score val set, save threshold_candidates to metadata pkl',
    )
    parser.set_defaults(**config)

    args = parser.parse_args(remaining_argv)
    args.config = config

    return args

def plot_likelihood_distributions(
    model,
    train_data,
    val_data,
    ood_data=None,
    thr= None,
    title="Likelihood Distribution",
    savepath=None,
    bins=50
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        train_data: np.ndarray or torch.Tensor, in-distribution training set
        val_data:   np.ndarray or torch.Tensor, held-out validation set
        ood_data:   np.ndarray or torch.Tensor, optional OOD dataset
        title: str, title for the plot
        savepath: str, optional path to save figure
        bins: int, number of histogram bins
    """
    # --- Compute log-likelihoods ---
    logp_train = model.score_samples(train_data).detach().cpu().numpy()
    logp_val   = model.score_samples(val_data).detach().cpu().numpy()
    logp_ood   = None
    if ood_data is not None:
        logp_ood = model.score_samples(ood_data).detach().cpu().numpy()


    # --- Plot ---
    plt.figure(figsize=(8, 5))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')
    plt.xlabel("Log-likelihood", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/train_distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/train_distribution_kde.png")
    plt.figure(figsize=(8, 5))


    if logp_ood is not None:
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="Test", kde=True)
        plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')

    plt.xlabel("Log-likelihood", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/distribution_kde.png")


def plot_val_test_id_distribution(
    model,
    val_data,
    test_id_data,
    thr=None,
    title="Validation vs Test ID Log-Likelihood Distribution",
    savepath="figures/val_test_id_distribution.png",
    bins=50
):
    """
    Visualize log-likelihood distributions for validation and test ID data in one plot.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        val_data: np.ndarray or torch.Tensor, validation set
        test_id_data: np.ndarray or torch.Tensor, in-distribution test set
        thr: float, threshold value to indicate on the plot
        title: str, title for the plot
        savepath: str, path to save figure
        bins: int, number of histogram bins
    """
    # --- Compute log-likelihoods ---
    logp_val = model.score_samples(val_data).detach().cpu().numpy()
    logp_test_id = model.score_samples(test_id_data).detach().cpu().numpy()

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.5, label="Validation", kde=True, stat="density")
    sns.histplot(logp_test_id, bins=bins, color="blue", alpha=0.5, label="Test ID", kde=True, stat="density")

    if thr is not None:
        plt.axvline(x=thr, color='tab:red', linestyle='--', linewidth=2, label=f'Threshold ({thr:.3f})')

    plt.xlabel("Log-Likelihood", fontsize=16, labelpad=10)
    plt.ylabel("Density", fontsize=16, labelpad=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.legend(fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2.5)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved figure at {savepath}")
    plt.close()

    # Print statistics
    print(f"\nValidation Log-Likelihood Statistics:")
    print(f"  Mean: {logp_val.mean():.4f}")
    print(f"  Std:  {logp_val.std():.4f}")
    print(f"  Min:  {logp_val.min():.4f}")
    print(f"  Max:  {logp_val.max():.4f}")

    print(f"\nTest ID Log-Likelihood Statistics:")
    print(f"  Mean: {logp_test_id.mean():.4f}")
    print(f"  Std:  {logp_test_id.std():.4f}")
    print(f"  Min:  {logp_test_id.min():.4f}")
    print(f"  Max:  {logp_test_id.max():.4f}")

    if thr is not None:
        val_below_thr = (logp_val < thr).sum() / len(logp_val) * 100
        test_below_thr = (logp_test_id < thr).sum() / len(logp_test_id) * 100
        print(f"\nPercentage below threshold:")
        print(f"  Validation: {val_below_thr:.2f}%")
        print(f"  Test ID:    {test_below_thr:.2f}%")

def create_ood_test(data, model, percentage=[0.1, 0.3, 0.5, 0.7, 0.9], noise_std=0.1):
    """
    Create OOD test sets with different OOD ratios and compute ROC AUC metrics.

    Args:
        data: torch.Tensor or np.ndarray, assumed to be (mostly) ID data
        model: trained RealNVP model with .predict and .score_samples
        percentage: list of OOD fractions (between 0 and 1)
        noise_std: standard deviation of Gaussian noise to create OOD samples

    Returns:
        data_dict: dict[perc] -> torch.FloatTensor test set on model.device
        res_dict:  dict[perc] -> dict with metrics (mean_score, roc_auc, accuracy, id_accuracy, ood_accuracy)
    """
    data_dict = {}
    res_dict = {}

    # Make a CPU tensor for all indexing / numpy ops
    if isinstance(data, torch.Tensor):
        data_cpu = data.detach().cpu()
    else:
        data_cpu = torch.as_tensor(data, dtype=torch.float32)

    # Get model device from parameters
    model_device = next(model.parameters()).device

    # Get inlier predictions (1 = normal, -1 = anomaly), using model's device internally
    predictions_tr = model.predict(data)  # returns numpy array

    # Indices of in-distribution points
    idx_in = np.where(predictions_tr == 1)[0]
    if len(idx_in) == 0:
        raise RuntimeError("No in-distribution points (predictions == 1) found in data.")

    # Take a 20% random subset of inliers as our base pool
    subset_size = max(1, int(0.2 * len(idx_in)))
    chosen_in = np.random.choice(idx_in, size=subset_size, replace=False)
    small_data = data_cpu[chosen_in].numpy()   # shape: (subset_size, D)

    # For each OOD percentage, build a mixed test set
    for perc in percentage:
        # perc = fraction of OOD samples in the test set
        test_size = subset_size
        n_ood = int(perc * test_size)
        n_id = test_size - n_ood

        # ID part: sample from small_data
        if n_id > 0:
            id_idx = np.random.choice(test_size, size=n_id, replace=False)
            id_part = small_data[id_idx]
        else:
            id_part = np.empty((0, small_data.shape[1]), dtype=small_data.dtype)

        # OOD part: noisy copies of small_data
        if n_ood > 0:
            ood_idx = np.random.choice(test_size, size=n_ood, replace=True)
            ood_base = small_data[ood_idx]
            noisy_part = ood_base + np.random.normal(0, noise_std, ood_base.shape)
        else:
            noisy_part = np.empty((0, small_data.shape[1]), dtype=small_data.dtype)

        mixed = np.concatenate([id_part, noisy_part], axis=0)

        # Move mixed test set onto model device for scoring
        mixed_tensor = torch.FloatTensor(mixed).to(model_device)
        data_dict[perc] = mixed_tensor

        # Create ground truth labels (0 = ID/normal, 1 = OOD/anomaly)
        y_true = np.concatenate([
            np.zeros(n_id),  # ID samples
            np.ones(n_ood)   # OOD samples
        ])

        # Score samples
        with torch.no_grad():
            scores_test = model.score_samples(mixed_tensor.to(model.device)).cpu().numpy()

        mean_score = scores_test.mean()

        # Compute ROC AUC if we have both classes
        if n_id > 0 and n_ood > 0:
            # Use negative log prob as anomaly score (higher = more anomalous)
            y_scores = -scores_test
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = None

        # Compute accuracy metrics using model's threshold
        if model.threshold is not None:
            # Predictions: log_prob < threshold means anomaly
            predictions = scores_test < model.threshold

            # Overall accuracy
            accuracy = (predictions == y_true).mean() if len(y_true) > 0 else None

            # ID accuracy (correctly classified as normal)
            if n_id > 0:
                id_predictions = scores_test[:n_id] >= model.threshold
                id_accuracy = id_predictions.mean()
            else:
                id_accuracy = None

            # OOD accuracy (correctly classified as anomaly)
            if n_ood > 0:
                ood_predictions = scores_test[n_id:] < model.threshold
                ood_accuracy = ood_predictions.mean()
            else:
                ood_accuracy = None
        else:
            accuracy = None
            id_accuracy = None
            ood_accuracy = None

        # Store comprehensive results
        res_dict[perc] = {
            'mean_score': mean_score,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'id_accuracy': id_accuracy,
            'ood_accuracy': ood_accuracy,
            'n_id': n_id,
            'n_ood': n_ood
        }

        print(f"Percentage OOD: {perc:.1%}")
        print(f"  Mean Score: {mean_score:.4f}")
        if roc_auc is not None:
            print(f"  ROC AUC: {roc_auc:.4f}")
        if accuracy is not None:
            print(f"  Overall Accuracy: {accuracy:.4f}")
        if id_accuracy is not None:
            print(f"  ID Accuracy: {id_accuracy:.4f}")
        if ood_accuracy is not None:
            print(f"  OOD Accuracy: {ood_accuracy:.4f}")
        print()

    return data_dict, res_dict


def create_ood_test_multiple_noise_levels(
    data,
    model,
    percentage=[0.1, 0.3, 0.5, 0.7, 0.9],
    noise_levels=[0.05, 0.1, 0.5]
):
    """
    Create multiple OOD test sets with different noise levels.

    Args:
        data: torch.Tensor or np.ndarray, assumed to be (mostly) ID data
        model: trained RealNVP model with .predict and .score_samples
        percentage: list of OOD fractions (between 0 and 1)
        noise_levels: list of noise standard deviations for creating OOD samples

    Returns:
        results_by_noise: dict[noise_std] -> {
            'data_dict': dict[perc] -> torch.FloatTensor test set,
            'res_dict': dict[perc] -> dict with metrics
        }
    """
    results_by_noise = {}

    for noise_std in noise_levels:
        print(f"\n{'='*80}")
        print(f"Creating OOD test sets with noise level: {noise_std}")
        print(f"{'='*80}")

        data_dict, res_dict = create_ood_test(
            data=data,
            model=model,
            percentage=percentage,
            noise_std=noise_std
        )

        results_by_noise[noise_std] = {
            'data_dict': data_dict,
            'res_dict': res_dict
        }

    return results_by_noise


def plot_ood_metrics(res_dict, test_id_score=None, save_dir="figures"):
    """
    Plot comprehensive OOD detection metrics.

    Args:
        res_dict: Dictionary with OOD percentages as keys and metrics as values
        test_id_score: Optional mean score for pure ID test data (0% OOD)
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data including pure ID test (0% OOD) if provided
    percentages = sorted(res_dict.keys())
    if test_id_score is not None:
        percentages = [0.0] + percentages

    mean_scores = []
    roc_aucs = []
    accuracies = []
    id_accuracies = []
    ood_accuracies = []

    for perc in percentages:
        if perc == 0.0 and test_id_score is not None:
            # Pure ID test data
            mean_scores.append(test_id_score)
            roc_aucs.append(None)  # No ROC AUC for pure ID data
            accuracies.append(None)
            id_accuracies.append(None)
            ood_accuracies.append(None)
        else:
            metrics = res_dict[perc]
            mean_scores.append(metrics['mean_score'])
            roc_aucs.append(metrics['roc_auc'])
            accuracies.append(metrics['accuracy'])
            id_accuracies.append(metrics['id_accuracy'])
            ood_accuracies.append(metrics['ood_accuracy'])

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OOD Detection Performance Metrics', fontsize=18, fontweight='bold', y=0.995)

    # Plot 1: Mean Log-Likelihood vs OOD Ratio
    ax1 = axes[0, 0]
    ax1.plot([p * 100 for p in percentages], mean_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('OOD Percentage (%)', fontsize=14)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=14)
    ax1.set_title('Log-Likelihood vs OOD Ratio', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([p * 100 for p in percentages])
    ax1.tick_params(labelsize=12)

    # Plot 2: ROC AUC vs OOD Ratio
    ax2 = axes[0, 1]
    valid_percentages = [p for p, auc in zip(percentages, roc_aucs) if auc is not None]
    valid_aucs = [auc for auc in roc_aucs if auc is not None]
    if valid_aucs:
        ax2.plot([p * 100 for p in valid_percentages], valid_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
        ax2.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax2.set_ylabel('ROC AUC', fontsize=14)
        ax2.set_title('ROC AUC vs OOD Ratio', fontsize=15, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_xticks([p * 100 for p in valid_percentages])
        ax2.tick_params(labelsize=12)

    # Plot 3: Accuracy Metrics vs OOD Ratio
    ax3 = axes[1, 0]
    valid_percentages_acc = [p for p, acc in zip(percentages, accuracies) if acc is not None]
    valid_accuracies = [acc for acc in accuracies if acc is not None]
    valid_id_acc = [acc for acc in id_accuracies if acc is not None]
    valid_ood_acc = [acc for acc in ood_accuracies if acc is not None]

    if valid_accuracies:
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_accuracies, 'o-', linewidth=2, markersize=8, label='Overall Accuracy', color='purple')
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_id_acc, 's-', linewidth=2, markersize=8, label='ID Accuracy', color='blue')
        ax3.plot([p * 100 for p in valid_percentages_acc], valid_ood_acc, '^-', linewidth=2, markersize=8, label='OOD Accuracy', color='red')
        ax3.set_xlabel('OOD Percentage (%)', fontsize=14)
        ax3.set_ylabel('Accuracy', fontsize=14)
        ax3.set_title('Accuracy Metrics vs OOD Ratio', fontsize=15, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=12)
        ax3.set_xticks([p * 100 for p in valid_percentages_acc])
        ax3.tick_params(labelsize=12)

    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = []
    headers = ['OOD %', 'Log-Like', 'ROC AUC', 'Accuracy', 'ID Acc', 'OOD Acc']

    for i, perc in enumerate(percentages):
        row = [f"{perc:.1%}"]
        row.append(f"{mean_scores[i]:.3f}")
        row.append(f"{roc_aucs[i]:.3f}" if roc_aucs[i] is not None else "N/A")
        row.append(f"{accuracies[i]:.3f}" if accuracies[i] is not None else "N/A")
        row.append(f"{id_accuracies[i]:.3f}" if id_accuracies[i] is not None else "N/A")
        row.append(f"{ood_accuracies[i]:.3f}" if ood_accuracies[i] is not None else "N/A")
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'ood_metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics summary plot to: {save_path}")
    plt.close()


def plot_roc_curves(model, test_ood_dict, res_ood_dict, save_dir="figures"):
    """
    Plot ROC curves for different OOD ratios.

    Args:
        model: Trained RealNVP model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        res_ood_dict: Dictionary with metrics including ground truth labels
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_ood_dict)))

    for i, (perc, test_data) in enumerate(sorted(test_ood_dict.items())):
        metrics = res_ood_dict[perc]
        n_id = metrics['n_id']
        n_ood = metrics['n_ood']

        if n_id > 0 and n_ood > 0:
            # Get scores
            with torch.no_grad():
                scores = model.score_samples(test_data.to(model.device)).cpu().numpy()

            # Create labels
            y_true = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
            y_scores = -scores  # Negative log prob as anomaly score

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, linewidth=2, color=colors[i],
                    label=f'{perc:.1%} OOD (AUC = {roc_auc:.3f})')

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)

    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('ROC Curves for Different OOD Ratios', fontsize=17, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(labelsize=13)
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(save_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves comparison to: {save_path}")
    plt.close()


def plot_likelihood_histograms(model, test_ood_dict, test_id_data=None, save_dir="figures"):
    """
    Plot likelihood distributions for different OOD ratios.

    Args:
        model: Trained RealNVP model
        test_ood_dict: Dictionary with OOD percentages as keys and test data as values
        test_id_data: Optional pure ID test data
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Log-Likelihood Distributions for Different OOD Ratios', fontsize=18, fontweight='bold', y=0.995)

    axes = axes.flatten()
    percentages = sorted(test_ood_dict.keys())

    # Add pure ID test if available
    if test_id_data is not None:
        with torch.no_grad():
            id_scores = model.score_samples(test_id_data.to(model.device)).cpu().numpy()

        ax = axes[0]
        ax.hist(id_scores, bins=50, color='blue', alpha=0.6, label='ID (0% OOD)', edgecolor='black')
        if model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'0% OOD (Pure ID)\nMean: {id_scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        plot_idx = 1
    else:
        plot_idx = 0

    # Plot for each OOD percentage
    for perc in percentages:
        if plot_idx >= len(axes):
            break

        test_data = test_ood_dict[perc]
        with torch.no_grad():
            scores = model.score_samples(test_data.to(model.device)).cpu().numpy()

        ax = axes[plot_idx]
        ax.hist(scores, bins=50, color='purple', alpha=0.6, edgecolor='black')
        if model.threshold is not None:
            ax.axvline(x=model.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Log-Likelihood', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(f'{perc:.1%} OOD\nMean: {scores.mean():.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(pad=2.5)
    save_path = os.path.join(save_dir, 'likelihood_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved likelihood histograms to: {save_path}")
    plt.close()


def plot_tsne(tsne_data1, preds, title):

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data1[preds == 1, 0], tsne_data1[preds == 1, 1],
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data1[preds == -1, 0], tsne_data1[preds == -1, 1],
                color='red', label='OOD', alpha=0.5)
    plt.title(title, fontsize=16, fontweight='bold')

    plt.xlabel('TSNE Dimension 1', fontsize=14)
    plt.ylabel('TSNE Dimension 2', fontsize=14)
    plt.tick_params(labelsize=12)
    #save figure
    plt.legend(fontsize=12)
    plt.tight_layout(pad=2.0)
    plt.savefig(f"figures/{title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.obs_dim = (args.obs_dim,)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = args.config

    # Override config with command line arguments if provided
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.verbose:
        config['verbose'] = True

    # Override config with RL-specific arguments
    if args.data_path is not None:
        args.data_path = args.data_path  # Keep as args attribute for compatibility
    if args.task != 'synthetic':
        args.task = args.task
    args.obs_shape = tuple(args.obs_dim)
    args.action_dim = args.action_dim

    # Set random seed for reproducibility: CLI overrides config
    if args.seed is not None:
        config['seed'] = args.seed
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    # Determine device: CLI overrides config
    cli_device = args.device
    cfg_device = config.get('device', None)

    device = cli_device if cli_device is not None else (cfg_device or 'cpu')

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    # Make sure everyone agrees on the same device
    args.device = device
    config["device"] = device

    print(f"Using device: {device}")


    # Choose data loading mode
    use_rl_data = config.get('use_rl_data', args.task != 'synthetic')

    # Only create environment if needed (when data_path is not provided)
    env = None
    if use_rl_data and args.data_path is None:
        try:
            env = gym.make(args.task)
        except Exception as e:
            print(f"Warning: Could not create environment {args.task}: {e}")
            print("Will use data_path instead if provided")

    if use_rl_data:
        print("Loading RL dataset for KDE training...")

        # Load RL data (next_observations + actions)
        train_data, val_data, test_data, kde_input_dim = load_rl_data_for_kde(
            args=args,
            env=env,  
            val_split_ratio=config.get('val_ratio', 0.2),
            test_split_ratio=config.get('test_ratio', 0.2)
        )

        # For RL data, we don't have separate anomaly data for evaluation
        # We'll use a portion of validation data as "normal" and generate synthetic anomalies
        n_test = len(test_data)
        test_normal = test_data

        # Generate synthetic anomalies in the same dimension as RL data
        anomaly_data = torch.randn(n_test // 2, kde_input_dim) * 3 + 5  # Offset anomalies

        # Update input dimension in config
        config['input_dim'] = kde_input_dim

    else:
        print("Creating synthetic data...")
        normal_data, anomaly_data = create_synthetic_data(
            n_samples=config.get('n_samples', 2000),
            dim=config.get('input_dim', 2),
            anomaly_type=config.get('anomaly_type', 'outlier')
        )
        print(anomaly_data.shape)
        # # Split normal data into train/val/test
        # train_ratio = config.get('train_ratio', 0.6)
        # val_ratio = config.get('val_ratio', 0.2)

        # n_train = int(train_ratio * len(normal_data))
        # n_val = int(val_ratio * len(normal_data))

        # train_data = normal_data[:n_train]
        # val_data = normal_data[n_train:n_train+n_val]
        # test_normal = normal_data[n_train+n_val:]

    if config.get('verbose', True):
        print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, "
              f"Test Normal: {test_normal.shape}, Test Anomaly: {anomaly_data.shape}")

    # --compute_thresholds_only: load existing model, score val set, save threshold_candidates
    if getattr(args, 'compute_thresholds_only', False):
        save_path = args.model_save_path if hasattr(args, 'model_save_path') and args.model_save_path else \
            config.get('model_save_path')
        if not save_path:
            raise ValueError("--compute_thresholds_only requires --model_save_path or model_save_path in config")

        print(f"\nLoading existing RealNVP model from: {save_path}")
        model_dict = RealNVP.load_model(save_path, device=device)
        rvp_model = model_dict['model']

        print("Scoring validation set...")
        with torch.no_grad():
            val_scores = rvp_model.score_samples(val_data.to(device))

        percentiles = [1, 5, 10, 15, 20]
        threshold_candidates = {p: float(np.percentile(val_scores, p)) for p in percentiles}
        print(f"Threshold candidates: {threshold_candidates}")

        meta_path = f"{save_path}_meta_data.pkl"
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        metadata['threshold_candidates'] = threshold_candidates
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved threshold_candidates to {meta_path}")
        sys.exit(0)

    # Create and train model
    print("Creating RealNVP model...")
    model = RealNVP(
        input_dim=config.get('input_dim', 2),
        num_layers=config.get('num_layers', 6),
        hidden_dims=config.get('hidden_dims', [256, 256]),
        device=device
    ).to(device)

    print("Training RealNVP model...")
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=config.get('epochs', 100),
        batch_size=config.get('batch_size', 128),
        lr=config.get('lr', 1e-3),
        patience=config.get('patience', 15),
        verbose=config.get('verbose', True)
    )
    # Save model if requested - use command-line arg if provided, otherwise use config
    save_path = None
    if hasattr(args, 'model_save_path') and args.model_save_path is not None:
        # Command line argument was explicitly provided
        save_path = args.model_save_path
    elif config.get('model_save_path', False):
        # Use config file value
        save_path = config.get('model_save_path', 'saved_models/realnvp')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_model(save_path)
        print(f"Model saved to: {save_path}_model.pth")
    # #load pretrained model (commented out - appears to be test/debug code)
    # model_dict = RealNVP.load_model(save_path=args.model_save_path)
    # model = model_dict['model']
    # model.to(device)
    # print(args.device)
    # model.threshold = model_dict['thr']
    # print(model.device)
    # Evaluate anomaly detection
    # print("\nEvaluating anomaly detection performance...")
    # results = model.evaluate_anomaly_detection(
    #     normal_data=test_normal.to(model.device),
    #     anomaly_data=anomaly_data.to(model.device),
    #     plot=config.get('plot_results', True)
    # )
    # print(type(train_data))
    # train_data = train_data.to(model.device)
    # val_data = val_data.to(model.device)
    # test_data = test_data.to(model.device)
    # print(train_data.device)
    # print(model.device)

    # predictions_tr = model.predict(train_data)
    # scores_tr = model.score_samples(train_data)
    # print('TRAINING SCORE', scores_tr.mean().item())
    # scores_test_in_dist = model.score_samples(test_data)
    # anomaly_test_res  = model.score_samples(anomaly_data.to(model.device))

    # small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    # noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    # normal_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(model.device)

    # print("\n" + "="*80)
    # print("OOD Test Results with Different Noise Levels:")
    # print("="*80)

    # # Test with 3 different noise levels
    # noise_levels = [0.05, 0.1, 0.5]  # Low, Medium, High noise
    # results_by_noise = create_ood_test_multiple_noise_levels(
    #     data=train_data,
    #     model=model,
    #     percentage=[0.1, 0.3, 0.5, 0.7, 0.9],
    #     noise_levels=noise_levels
    # )

  
    # Plot validation vs test ID distribution
    # print("\n" + "="*80)
    # print("Creating Validation vs Test ID Distribution Plot...")
    # print("="*80 + "\n")
    # plot_val_test_id_distribution(
    #     model=model,
    #     val_data=val_data,
    #     test_id_data=test_data,
    #     thr=model.threshold,
    #     title="Validation vs Test ID Log-Likelihood Distribution",
    #     savepath=f"figures/{args.task}/val_test_id_distribution.png",
    #     bins=50
    # )

    # Generate comprehensive plots
    # print("\n" + "="*80)
    # print("Generating Visualization Plots...")
    # print("="*80 + "\n")

    # Plot comparison across all noise levels
    # compare_noise_levels(
    #     results_by_noise=results_by_noise,
    #     test_id_score=scores_test_in_dist.mean().item(),
    #     save_dir=f"figures/{args.task}"
    # )

    # Generate individual plots for each noise level
    # for noise_std in noise_levels:
    #     print(f"\nGenerating plots for noise level σ={noise_std}...")
    #     test_ood_dict = results_by_noise[noise_std]['data_dict']
    #     res_ood_dict = results_by_noise[noise_std]['res_dict']

    #     # Create subdirectory for this noise level
    #     noise_dir = f"figures/{args.task}/noise_{noise_std}"
    #     os.makedirs(noise_dir, exist_ok=True)

    #     # Plot 1: Comprehensive metrics summary (likelihood, accuracy, AUC)
    #     plot_ood_metrics(
    #         res_dict=res_ood_dict,
    #         test_id_score=scores_test_in_dist.mean().item(),
    #         save_dir=noise_dir
    #     )

    #     # Plot 2: ROC curves comparison
    #     plot_roc_curves(
    #         model=model,
    #         test_ood_dict=test_ood_dict,
    #         res_ood_dict=res_ood_dict,
    #         save_dir=noise_dir
    #     )

    #     # Plot 3: Likelihood histograms for all OOD ratios
    #     plot_likelihood_histograms(
    #         model=model,
    #         test_ood_dict=test_ood_dict,
    #         test_id_data=test_data,
    #         save_dir=noise_dir
    #     )

    # # Original likelihood distribution plot
    # plot_likelihood_distributions(
    #     model,
    #     train_data,
    #     val_data,
    #     ood_data=normal_data,
    #     thr=model.threshold,
    #     title="Likelihood Distribution",
    #     savepath=args.fig_save_path,
    #     bins=50
    # )

    # print("\n" + "="*80)
    # print("All plots saved to 'figures/' directory")
    # print("="*80)
    # # print(f"ROC AUC: {results['roc_auc']:.3f}")
    # # print(f"Accuracy: {results['accuracy']:.3f}")
    # # print(f"Normal data log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    # # print(f"Anomaly data log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")


    

    print("\nRealNVP training and evaluation completed!")