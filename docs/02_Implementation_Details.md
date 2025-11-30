# Implementation Details
## Context-Guided Multi-Scale LNN - Complete Code Guide

**Purpose:** Production-ready implementation with best practices

---

## 1. Project Structure

```
capstone/
├── config/
│   ├── config.yaml              # Hyperparameters & paths
│   └── experiment_config.py     # Experiment configurations
├── data/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset class
│   ├── preprocessing.py         # Feature extraction & normalization
│   └── augmentation.py          # Data augmentation (optional)
├── models/
│   ├── __init__.py
│   ├── context_ltc_cell.py      # Core LTC cell with context modulation
│   ├── multi_scale_lnn.py       # Complete model architecture
│   └── baselines.py             # Baseline models (LSTM, CNN, LogReg)
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop & validation
│   ├── losses.py                # Custom loss functions
│   └── metrics.py               # Evaluation metrics
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging utilities
│   ├── visualization.py         # Plot training curves, predictions
│   └── checkpointing.py         # Save/load model checkpoints
├── experiments/
│   ├── train_multi_scale_lnn.py # Main training script
│   ├── evaluate.py              # Model evaluation script
│   └── ablation_study.py        # Ablation experiments
├── notebooks/
│   └── analysis.ipynb           # Result analysis & visualization
├── dataset/
│   └── merged_cir.csv           # Your data
├── eda/
│   └── eda.ipynb                # Your EDA notebook
└── requirements.txt             # Dependencies
```

---

## 2. Core Implementation Files

### 2.1 Context-LTC Cell (`models/context_ltc_cell.py`)

```python
"""
Context-Guided Liquid Time-Constant Cell
Implements adaptive τ modulation based on domain knowledge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextLTCCell(nn.Module):
    """
    Liquid Time-Constant cell with context-driven tau modulation.
    
    Dynamics:
        dx/dt = (-x + f(W_in·u + W_rec·x + b)) / τ(context)
        x_next = x + dt · (dx/dt)
    
    Where τ(context) is dynamically computed from domain features.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        context_size: int = 7,
        tau_base: float = 1e-9,  # Base tau in seconds (e.g., 1 ns)
        tau_range: tuple = (0.5, 2.0),  # Modulation range (multipliers)
        dt: float = 15.65e-12,  # Integration timestep (15.65 ps per CIR sample)
        activation: str = 'tanh'
    ):
        """
        Args:
            input_size: Dimension of input (usually 1 for CIR amplitude)
            hidden_size: Number of hidden units
            context_size: Dimension of context feature vector
            tau_base: Base time constant in seconds
            tau_range: (min_mult, max_mult) for tau modulation
            dt: Integration timestep (should match CIR sampling period)
            activation: Nonlinearity ('tanh', 'relu', 'gelu')
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.tau_base = tau_base
        self.tau_min, self.tau_max = tau_range
        self.dt = dt
        
        # Input transformation
        self.W_in = nn.Linear(input_size, hidden_size)
        
        # Recurrent transformation
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Context → Tau gate (learnable mapping)
        self.tau_gate = nn.Sequential(
            nn.Linear(context_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),  # Single scalar per sample
            nn.Sigmoid()  # Maps to [0, 1]
        )
        
        # Activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def compute_tau(self, context):
        """
        Compute dynamic tau from context features.
        
        Args:
            context: (batch_size, context_size)
        
        Returns:
            tau_effective: (batch_size, 1) in seconds
        """
        # Context → [0, 1] gate
        gate = self.tau_gate(context)  # (B, 1)
        
        # Map [0,1] → [tau_min, tau_max] multiplier
        multiplier = self.tau_min + (self.tau_max - self.tau_min) * gate
        
        # Apply to base tau
        tau_effective = self.tau_base * multiplier  # (B, 1)
        
        return tau_effective
    
    def forward(self, u_t, h_t, context):
        """
        Single timestep forward pass.
        
        Args:
            u_t: Input at time t, shape (batch_size, input_size)
            h_t: Hidden state at time t, shape (batch_size, hidden_size)
            context: Context features, shape (batch_size, context_size)
        
        Returns:
            h_next: Updated hidden state, shape (batch_size, hidden_size)
            tau: Effective tau used, shape (batch_size, 1)
        """
        batch_size = u_t.size(0)
        
        # Compute dynamic tau from context
        tau = self.compute_tau(context)  # (B, 1)
        
        # Standard LTC dynamics
        z = self.W_in(u_t) + self.W_rec(h_t) + self.bias  # (B, H)
        f = self.activation(z)  # (B, H)
        
        # ODE: dx/dt = (-x + f) / tau
        dh_dt = (-h_t + f) / tau  # (B, H) / (B, 1) → broadcasts
        
        # Euler integration: h_next = h_t + dt * dh_dt
        h_next = h_t + self.dt * dh_dt  # (B, H)
        
        return h_next, tau


class ContextLTCLayer(nn.Module):
    """
    Wrapper to unroll ContextLTCCell over a full sequence.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        context_size: int = 7,
        tau_base: float = 1e-9,
        tau_range: tuple = (0.5, 2.0),
        dt: float = 15.65e-12,
        activation: str = 'tanh'
    ):
        super().__init__()
        
        self.cell = ContextLTCCell(
            input_size=input_size,
            hidden_size=hidden_size,
            context_size=context_size,
            tau_base=tau_base,
            tau_range=tau_range,
            dt=dt,
            activation=activation
        )
        self.hidden_size = hidden_size
    
    def forward(self, seq, context):
        """
        Unroll over sequence.
        
        Args:
            seq: Input sequence, shape (batch_size, seq_len, input_size)
            context: Context features, shape (batch_size, context_size)
        
        Returns:
            outputs: All hidden states, shape (batch_size, seq_len, hidden_size)
            h_final: Final hidden state, shape (batch_size, hidden_size)
            tau_history: Tau values used, shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = seq.size()
        device = seq.device
        
        # Initialize hidden state
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        tau_history = []
        
        for t in range(seq_len):
            u_t = seq[:, t, :]  # (B, input_size)
            h_t, tau = self.cell(u_t, h_t, context)
            outputs.append(h_t)
            tau_history.append(tau)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (B, T, H)
        tau_history = torch.stack(tau_history, dim=1)  # (B, T, 1)
        h_final = h_t  # (B, H)
        
        return outputs, h_final, tau_history


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    # Test ContextLTCCell
    print("Testing ContextLTCCell...")
    
    batch_size = 4
    input_size = 1
    hidden_size = 32
    context_size = 7
    
    cell = ContextLTCCell(
        input_size=input_size,
        hidden_size=hidden_size,
        context_size=context_size,
        tau_base=1e-9,  # 1 ns
        tau_range=(0.5, 2.0)
    )
    
    u_t = torch.randn(batch_size, input_size)
    h_t = torch.randn(batch_size, hidden_size)
    context = torch.rand(batch_size, context_size)  # Normalized [0,1]
    
    h_next, tau = cell(u_t, h_t, context)
    
    print(f"Input shape: {u_t.shape}")
    print(f"Hidden state shape: {h_t.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output hidden shape: {h_next.shape}")
    print(f"Tau shape: {tau.shape}")
    print(f"Tau range: [{tau.min().item():.2e}, {tau.max().item():.2e}] seconds")
    
    # Test ContextLTCLayer
    print("\nTesting ContextLTCLayer...")
    
    layer = ContextLTCLayer(
        input_size=input_size,
        hidden_size=hidden_size,
        context_size=context_size,
        tau_base=1e-9
    )
    
    seq = torch.randn(batch_size, 1016, input_size)  # Full CIR sequence
    context = torch.rand(batch_size, context_size)
    
    outputs, h_final, tau_history = layer(seq, context)
    
    print(f"Sequence shape: {seq.shape}")
    print(f"Outputs shape: {outputs.shape}")
    print(f"Final hidden shape: {h_final.shape}")
    print(f"Tau history shape: {tau_history.shape}")
    
    print("\n✅ ContextLTCCell tests passed!")
```

---

### 2.2 Multi-Scale LNN Model (`models/multi_scale_lnn.py`)

```python
"""
Multi-Scale Liquid Neural Network with Context Guidance
Three-tau architecture for UWB LOS/NLOS classification
"""

import torch
import torch.nn as nn
from models.context_ltc_cell import ContextLTCLayer


class MultiScaleLNN(nn.Module):
    """
    Context-Guided Multi-Scale Liquid Neural Network.
    
    Architecture:
        1. Three parallel LTC layers with different base tau:
           - Small-tau (50 ps): Captures rise dynamics
           - Medium-tau (1 ns): Captures first bounce
           - Large-tau (5 ns): Captures multipath tail
        
        2. Context features modulate each tau independently
        
        3. Fusion layer combines multi-scale features
        
        4. Dual output heads:
           - Classification: LOS/NLOS (binary)
           - Regression: Distance correction (meters)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        context_size: int = 7,
        tau_small: float = 50e-12,   # 50 ps
        tau_medium: float = 1e-9,    # 1 ns
        tau_large: float = 5e-9,     # 5 ns
        tau_range: tuple = (0.5, 2.0),
        dt: float = 15.65e-12,       # CIR sampling period
        dropout: float = 0.3
    ):
        """
        Args:
            input_size: CIR amplitude dimension (1)
            hidden_size: Hidden units per LTC layer
            context_size: Number of context features
            tau_small/medium/large: Base tau for each scale
            tau_range: Modulation range multiplier
            dt: Integration timestep
            dropout: Dropout probability for output heads
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        # ========================================
        # Three-Tau LTC Layers (Parallel)
        # ========================================
        
        self.lnn_small = ContextLTCLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            context_size=context_size,
            tau_base=tau_small,
            tau_range=tau_range,
            dt=dt,
            activation='tanh'
        )
        
        self.lnn_medium = ContextLTCLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            context_size=context_size,
            tau_base=tau_medium,
            tau_range=tau_range,
            dt=dt,
            activation='tanh'
        )
        
        self.lnn_large = ContextLTCLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            context_size=context_size,
            tau_base=tau_large,
            tau_range=tau_range,
            dt=dt,
            activation='tanh'
        )
        
        # ========================================
        # Fusion & Output Heads
        # ========================================
        
        fused_size = hidden_size * 3  # Concatenate 3 layers
        
        # Classification Head: LOS (0) vs NLOS (1)
        self.classifier = nn.Sequential(
            nn.Linear(fused_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability P(NLOS)
        )
        
        # Regression Head: Distance correction (meters)
        self.regressor = nn.Sequential(
            nn.Linear(fused_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Linear output for distance
        )
    
    def forward(self, cir_seq, context):
        """
        Forward pass through multi-scale LNN.
        
        Args:
            cir_seq: Raw CIR sequence, shape (batch_size, 1016, 1)
            context: Context features, shape (batch_size, 7)
        
        Returns:
            prob_nlos: P(NLOS), shape (batch_size, 1)
            distance: Predicted distance, shape (batch_size, 1)
            tau_dict: Dictionary of tau histories for analysis
        """
        # Run through three parallel LTC layers
        _, h_small, tau_small = self.lnn_small(cir_seq, context)
        _, h_medium, tau_medium = self.lnn_medium(cir_seq, context)
        _, h_large, tau_large = self.lnn_large(cir_seq, context)
        
        # Concatenate final hidden states
        h_fused = torch.cat([h_small, h_medium, h_large], dim=1)  # (B, 3*H)
        
        # Dual outputs
        prob_nlos = self.classifier(h_fused)  # (B, 1)
        distance = self.regressor(h_fused)    # (B, 1)
        
        # Return tau histories for analysis (optional)
        tau_dict = {
            'small': tau_small,
            'medium': tau_medium,
            'large': tau_large
        }
        
        return prob_nlos, distance, tau_dict
    
    def predict_class(self, cir_seq, context, threshold=0.5):
        """
        Predict discrete class labels.
        
        Args:
            cir_seq: Raw CIR, shape (batch_size, 1016, 1)
            context: Context features, shape (batch_size, 7)
            threshold: Classification threshold for P(NLOS)
        
        Returns:
            labels: Predicted labels (0=LOS, 1=NLOS), shape (batch_size,)
            prob_nlos: Probability scores, shape (batch_size, 1)
            distance: Predicted distance, shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            prob_nlos, distance, _ = self.forward(cir_seq, context)
            labels = (prob_nlos > threshold).float().squeeze()
        return labels, prob_nlos, distance


# ============================================================================
# Model Summary & Testing
# ============================================================================

if __name__ == "__main__":
    from torchsummary import summary
    
    print("Creating Multi-Scale LNN...")
    model = MultiScaleLNN(
        input_size=1,
        hidden_size=64,
        context_size=7,
        tau_small=50e-12,
        tau_medium=1e-9,
        tau_large=5e-9,
        dropout=0.3
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 8
    cir_seq = torch.randn(batch_size, 1016, 1)
    context = torch.rand(batch_size, 7)
    
    prob_nlos, distance, tau_dict = model(cir_seq, context)
    
    print(f"CIR sequence shape: {cir_seq.shape}")
    print(f"Context shape: {context.shape}")
    print(f"P(NLOS) shape: {prob_nlos.shape}")
    print(f"Distance shape: {distance.shape}")
    print(f"\nTau statistics:")
    for name, tau_hist in tau_dict.items():
        print(f"  {name}-tau: mean={tau_hist.mean().item():.2e}s, "
              f"std={tau_hist.std().item():.2e}s")
    
    print("\n✅ Multi-Scale LNN tests passed!")
```

---

### 2.3 Dataset & Preprocessing (`data/dataset.py`)

```python
"""
PyTorch Dataset for UWB CIR data with context feature extraction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class UWBCIRDataset(Dataset):
    """
    UWB Channel Impulse Response Dataset with automatic context feature extraction.
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        normalize_cir: bool = True,
        random_seed: int = 42
    ):
        """
        Args:
            csv_path: Path to merged_cir.csv
            split: 'train' or 'test'
            train_ratio: Fraction of data for training
            normalize_cir: If True, normalize CIR to [-1, 1]
            random_seed: For reproducible splitting
        """
        self.csv_path = csv_path
        self.split = split
        self.normalize_cir = normalize_cir
        
        # Load data
        print(f"Loading data from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        print(f"Total samples: {len(self.data)}")
        
        # Constants
        self.TS_DW1000 = 1 / (128 * 499.2e6)  # 15.65 ps
        self.FP_INDEX_SCALE = 64.0
        self.CIR_COLS = [f'CIR{i}' for i in range(1016)]
        
        # Extract context features
        print("Extracting context features...")
        self._extract_context_features()
        
        # Train/test split (stratified by label)
        self._split_data(train_ratio, random_seed)
        
        # Fit scalers on training data only
        if split == 'train':
            self._fit_scalers()
        
        print(f"{split.upper()} set: {len(self.indices)} samples")
    
    def _extract_context_features(self):
        """Extract all 7 context features from raw CIR."""
        data = self.data
        cir_cols = self.CIR_COLS
        
        # 1. FP_INDEX_scaled (hardware first path)
        data['FP_INDEX_scaled'] = data['FP_INDEX'] / self.FP_INDEX_SCALE
        
        # 2. Max_Index (peak location)
        cir_data = data[cir_cols].values
        data['Max_Index'] = np.argmax(np.abs(cir_data), axis=1)
        
        # 3. t_start (from hardware FP)
        data['t_start'] = data['FP_INDEX_scaled']
        
        # 4. t_peak (same as Max_Index)
        data['t_peak'] = data['Max_Index']
        
        # 5. Rise_Time & Rise_Time_ns
        data['Rise_Time'] = data['t_peak'] - data['t_start']
        data['Rise_Time_ns'] = data['Rise_Time'] * self.TS_DW1000 * 1e9
        
        # 6. RiseRatio
        def compute_rise_ratio(row):
            t_start = int(row['t_start']) if not pd.isna(row['t_start']) else None
            t_peak = int(row['t_peak']) if not pd.isna(row['t_peak']) else None
            
            if t_start is None or t_peak is None:
                return np.nan
            if t_start < 0 or t_start >= len(cir_cols):
                return np.nan
            if t_peak >= len(cir_cols):
                return np.nan
            
            amp_start = abs(row[f'CIR{t_start}'])
            amp_peak = abs(row[f'CIR{t_peak}'])
            
            if amp_peak == 0:
                return np.nan
            
            return amp_start / amp_peak
        
        data['RiseRatio'] = data.apply(compute_rise_ratio, axis=1)
        
        # 7. E_tail (tail energy ratio)
        def compute_e_tail(row):
            t_peak = int(row['t_peak']) if not pd.isna(row['t_peak']) else None
            if t_peak is None or t_peak >= len(cir_cols) - 50:
                return np.nan
            
            cir = row[cir_cols].values
            tail_start = t_peak
            tail_end = min(t_peak + 50, len(cir_cols))
            
            tail_energy = np.sum(cir[tail_start:tail_end] ** 2)
            total_energy = np.sum(cir ** 2)
            
            if total_energy == 0:
                return np.nan
            
            return tail_energy / total_energy
        
        data['E_tail'] = data.apply(compute_e_tail, axis=1)
        
        # 8. Peak_SNR
        def compute_peak_snr(row):
            noise_floor = np.median(np.abs(row[self.CIR_COLS[:600]]))
            t_peak = int(row['t_peak']) if not pd.isna(row['t_peak']) else None
            if t_peak is None or t_peak >= len(cir_cols):
                return np.nan
            peak_amp = abs(row[f'CIR{t_peak}'])
            if noise_floor == 0:
                return np.nan
            return peak_amp / noise_floor
        
        data['Peak_SNR'] = data.apply(compute_peak_snr, axis=1)
        
        # 9. multipath_count (simplified - count peaks > 5× noise)
        def count_multipath(row):
            cir = row[cir_cols].values
            noise = np.median(np.abs(cir[:600]))
            threshold = 5 * noise
            # Simple peak counting in ROI
            roi = cir[650:900]
            peaks = (roi[1:-1] > roi[:-2]) & (roi[1:-1] > roi[2:]) & (roi[1:-1] > threshold)
            return np.sum(peaks)
        
        data['multipath_count'] = data.apply(count_multipath, axis=1)
        
        # Drop rows with NaN context features
        context_features = ['Rise_Time_ns', 'RiseRatio', 'E_tail', 'Peak_SNR', 
                           'multipath_count', 't_start', 't_peak']
        data.dropna(subset=context_features, inplace=True)
        
        self.data = data
        self.context_feature_names = context_features
    
    def _split_data(self, train_ratio, random_seed):
        """Stratified train/test split."""
        np.random.seed(random_seed)
        
        los_indices = self.data[self.data['label'] == 'LOS'].index.tolist()
        nlos_indices = self.data[self.data['label'] == 'NLOS'].index.tolist()
        
        np.random.shuffle(los_indices)
        np.random.shuffle(nlos_indices)
        
        n_train_los = int(len(los_indices) * train_ratio)
        n_train_nlos = int(len(nlos_indices) * train_ratio)
        
        train_indices = los_indices[:n_train_los] + nlos_indices[:n_train_nlos]
        test_indices = los_indices[n_train_los:] + nlos_indices[n_train_nlos:]
        
        if self.split == 'train':
            self.indices = train_indices
        else:
            self.indices = test_indices
        
        np.random.shuffle(self.indices)
    
    def _fit_scalers(self):
        """Fit MinMaxScaler on training context features."""
        train_data = self.data.loc[self.indices]
        context_values = train_data[self.context_feature_names].values
        
        self.context_scaler = MinMaxScaler(feature_range=(0, 1))
        self.context_scaler.fit(context_values)
        
        # Save scaler for test set
        self.data.attrs['context_scaler'] = self.context_scaler
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            cir_seq: (1016, 1) tensor
            context: (7,) tensor
            label: 0 (LOS) or 1 (NLOS)
            distance: true distance in meters
        """
        data_idx = self.indices[idx]
        row = self.data.loc[data_idx]
        
        # Raw CIR sequence
        cir_seq = row[self.CIR_COLS].values.astype(np.float32)
        
        # Normalize CIR to [-1, 1]
        if self.normalize_cir:
            cir_max = np.abs(cir_seq).max()
            if cir_max > 0:
                cir_seq = cir_seq / cir_max
        
        cir_seq = torch.from_numpy(cir_seq).unsqueeze(-1)  # (1016, 1)
        
        # Context features
        context = row[self.context_feature_names].values.astype(np.float32)
        
        # Scale context to [0, 1]
        if hasattr(self, 'context_scaler'):
            context = self.context_scaler.transform(context.reshape(1, -1))[0]
        
        context = torch.from_numpy(context)  # (7,)
        
        # Label: LOS=0, NLOS=1
        label = 0.0 if row['label'] == 'LOS' else 1.0
        label = torch.tensor(label, dtype=torch.float32)
        
        # True distance
        distance = torch.tensor(row['d_true'], dtype=torch.float32)
        
        return cir_seq, context, label, distance


# ============================================================================
# DataLoader Creation
# ============================================================================

def create_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    random_seed: int = 42
):
    """
    Create train and test DataLoaders.
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = UWBCIRDataset(
        csv_path=csv_path,
        split='train',
        train_ratio=train_ratio,
        normalize_cir=True,
        random_seed=random_seed
    )
    
    test_dataset = UWBCIRDataset(
        csv_path=csv_path,
        split='test',
        train_ratio=train_ratio,
        normalize_cir=True,
        random_seed=random_seed
    )
    
    # Copy scaler from train to test
    test_dataset.context_scaler = train_dataset.context_scaler
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    csv_path = "../dataset/merged_cir.csv"
    
    train_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        batch_size=16,
        train_ratio=0.8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    cir_seq, context, label, distance = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  CIR sequence: {cir_seq.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Label: {label.shape}")
    print(f"  Distance: {distance.shape}")
    
    print("\n✅ Dataset tests passed!")
```

---

## 3. Training Script (Summary)

**File:** `experiments/train_multi_scale_lnn.py`

```python
"""
Main training script for Multi-Scale LNN
Run: python experiments/train_multi_scale_lnn.py --config config/config.yaml
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.multi_scale_lnn import MultiScaleLNN
from data.dataset import create_dataloaders
from training.trainer import Trainer
from utils.logger import setup_logger

def main():
    # Hyperparameters
    config = {
        'data_path': 'dataset/merged_cir.csv',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 100,
        'early_stopping_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        csv_path=config['data_path'],
        batch_size=config['batch_size']
    )
    
    # Create model
    model = MultiScaleLNN(
        input_size=1,
        hidden_size=64,
        context_size=7,
        tau_small=50e-12,
        tau_medium=1e-9,
        tau_large=5e-9,
        dropout=0.3
    ).to(config['device'])
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Loss functions
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        device=config['device'],
        loss_weights={'cls': 1.0, 'reg': 0.1}
    )
    
    # Train!
    trainer.train(epochs=config['epochs'])

if __name__ == "__main__":
    main()
```

---

## 4. Key Implementation Notes

### 4.1 Tau Value Selection
```python
# Based on your actual signal timescales:
tau_small = 50e-12    # 50 ps  ≈ 1-2× rise time (25-42 ps)
tau_medium = 1e-9     # 1 ns   ≈ first bounce timescale
tau_large = 5e-9      # 5 ns   ≈ 1/3 of total CIR duration (15.9 ns)
```

### 4.2 Context Feature Normalization
```python
# CRITICAL: Always normalize context features to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
context_normalized = scaler.fit_transform(context_features)
```

### 4.3 Loss Function Weighting
```python
# Classification loss more important than distance regression
L_total = 1.0 * L_classification + 0.1 * L_regression
```

### 4.4 Gradient Clipping
```python
# Prevent exploding gradients in LTC dynamics
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. Next Steps

1. **Run baseline experiments** (Logistic Regression, LSTM, CNN) - See `03_Comparison_Baselines.md`
2. **Implement ablation studies** - Remove context, single-tau, etc.
3. **Hyperparameter tuning** - Grid search on tau values, hidden size
4. **Visualization** - Plot tau evolution, attention maps

**Continue to:** `03_Comparison_with_Baselines.md` for detailed comparison framework!
