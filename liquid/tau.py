import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextLTCCell(nn.Module):
    """
    Context-Modulated LTC Cell.
    The time-constant 'tau' is DYNAMIC. It changes based on the 'context' (Rise_Time).
    
    Equation:
      tau_t = base_tau * (1 + sigmoid(W_ctx * context))
    """
    def __init__(self, input_size, hidden_size, context_size, base_tau=1.0, dt=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.base_tau = base_tau
        self.dt = dt
        
        # Standard RNN weights
        self.inp = nn.Linear(input_size, hidden_size)
        self.rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # --- THE LIQUID MODULATOR ---
        # Maps the context features (Rise Time, Energy) to the Tau gate
        self.tau_modulator = nn.Linear(context_size, hidden_size)

    def forward(self, u_t, x_t, context):
        # 1. Calculate the standard update candidate
        z = self.inp(u_t) + self.rec(x_t) + self.bias
        f = torch.tanh(z)
        
        # 2. Calculate Dynamic Tau based on Context
        # If Rise_Time is high -> Gate is high -> Tau increases (Sluggish/Memory mode)
        # If Rise_Time is low  -> Gate is low  -> Tau stays base (Fast/Direct mode)
        tau_gate = torch.sigmoid(self.tau_modulator(context)) 
        current_tau = self.base_tau * (1.0 + 5.0 * tau_gate) # Scale factor 5.0 allows tau to expand
        
        # 3. Apply Liquid Update (ODE Solver)
        dx_dt = (-x_t + f) / current_tau
        x_next = x_t + self.dt * dx_dt
        
        return x_next

class LiquidNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, context_size=3, fcn_size=64):
        super().__init__()
        
        # We only need ONE liquid layer now, because the Tau adapts automatically!
        # But we can still keep 3 parallel branches if you want "multi-scale" redundancy.
        # Let's stick to a robust single modulated layer for clarity.
        self.cell = ContextLTCCell(input_size, hidden_size, context_size, base_tau=1.0)
        self.hidden_size = hidden_size
        
        # Prediction Heads
        self.fcn = nn.Sequential(
            nn.Linear(hidden_size, fcn_size),
            nn.ReLU(),
            nn.Linear(fcn_size, fcn_size),
            nn.ReLU()
        )
        self.reg_head = nn.Linear(fcn_size, 1) # Predicts True Distance

    def forward(self, cir_sequence, context_features):
        """
        cir_sequence: (Batch, 1016, 1) -> The Waveform
        context_features: (Batch, 3)   -> [Rise_Time, Energy, Max_Amp]
        """
        B, T, D = cir_sequence.shape
        h = torch.zeros(B, self.hidden_size, device=cir_sequence.device)
        
        # Unroll over time
        for t in range(T):
            u_t = cir_sequence[:, t, :] # Current CIR value
            
            # Feed Context into the cell at every step to keep Tau modulated
            h = self.cell(u_t, h, context_features)
            
        # Final prediction based on the final state
        feat = self.fcn(h)
        dist_pred = self.reg_head(feat)
        
        return dist_pred