import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import FullyConnected

class MultiScaleLiquidNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Define 3 separate Liquid Layers (The "Buckets")
        # Each will learn its own internal dynamics
        wiring = FullyConnected(units=hidden_size) # Connectivity pattern
        
        # Branch 1: Fast / Small Tau (First Path)
        self.lnn_fast = LTC(input_size, wiring)
        
        # Branch 2: Medium Tau (Body)
        self.lnn_med = LTC(input_size, wiring)
        
        # Branch 3: Slow / Large Tau (Tail)
        self.lnn_slow = LTC(input_size, wiring)
        
        # The Output Heads (Classification & Regression)
        # Input to heads is 3x hidden_size (concatenated features from all 3 branches)
        combined_size = hidden_size * 3
        
        # Head 1: NLOS Classification (Sigmoid output 0-1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Distance Regression (Linear output)
        self.regressor = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, time, features) -> (Batch, 512, 2) for Real/Imag pairs
        
        # Run data through all 3 Liquid branches in parallel
        # Note: In a real implementation, you might initialize their time-constants 
        # manually here if the library defaults don't spread them out enough.
        out_fast, _ = self.lnn_fast(x)
        out_med, _  = self.lnn_med(x)
        out_slow, _ = self.lnn_slow(x)
        
        # We usually take the LAST state of the sequence for classification/regression
        # Shape becomes (Batch, Hidden_Size)
        feat_fast = out_fast[:, -1, :]
        feat_med  = out_med[:, -1, :]
        feat_slow = out_slow[:, -1, :]
        
        # Concatenate the "views" (Fast + Med + Slow)
        combined = torch.cat([feat_fast, feat_med, feat_slow], dim=1)
        
        # Generate Outputs
        prob_nlos = self.classifier(combined)
        dist_correction = self.regressor(combined)
        
        return prob_nlos, dist_correction

# Example Usage
model = MultiScaleLiquidNet(input_size=2, hidden_size=32, output_size=1)