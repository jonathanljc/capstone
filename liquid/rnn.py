import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCStyleCell(nn.Module):
    """
    Simple LTC-style cell:
      dx/dt = (-x + tanh(W_in u + W_rec x + b)) / tau
      x_next = x + dt * dx/dt
    """
    def __init__(self, input_size, hidden_size, tau=5.0, dt=1.0, learn_tau=False):
        super().__init__()
        self.inp = nn.Linear(input_size, hidden_size)
        self.rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        tau_tensor = torch.tensor(float(tau))
        self.tau = nn.Parameter(tau_tensor, requires_grad=learn_tau)  # fixed or learnable

        self.dt = dt

    def forward(self, u_t, x_t):
        # u_t: (B, input_size)
        # x_t: (B, hidden_size)
        z = self.inp(u_t) + self.rec(x_t) + self.bias
        f = torch.tanh(z)

        dx_dt = (-x_t + f) / self.tau
        x_next = x_t + self.dt * dx_dt
        return x_next


class LTCStyleLayer(nn.Module):
    """
    Unrolls one LTCStyleCell over the whole CIR sequence.
    """
    def __init__(self, input_size, hidden_size, tau=5.0, dt=1.0, learn_tau=False):
        super().__init__()
        self.cell = LTCStyleCell(input_size, hidden_size, tau, dt, learn_tau)
        self.hidden_size = hidden_size

    def forward(self, seq):
        # seq: (B, T, input_size)
        B, T, D = seq.shape
        h = torch.zeros(B, self.hidden_size, device=seq.device)
        outputs = []

        for t in range(T):
            u_t = seq[:, t, :]      # (B, D)
            h = self.cell(u_t, h)   # (B, H)
            outputs.append(h.unsqueeze(1))

        out_seq = torch.cat(outputs, dim=1)  # (B, T, H)
        h_last = out_seq[:, -1, :]          # (B, H)
        return out_seq, h_last


class MultiTauLTCNet(nn.Module):
    """
    Multi-scale liquid network for CIR-only input.

    - Input:  CIR magnitude sequence, shape (B, T, 1)
    - Branches:
        fast   tau=1   (sensitive to early / first-path part)
        medium tau=5   (good for single-bounce lobe)
        slow   tau=20  (integrates long multipath tail)
    - Heads:
        cls_head: single-bounce probability (0/1)
        reg_head: distance (or distance correction)
    """
    def __init__(self, input_size=1, hidden_size=32, fcn_size=64):
        super().__init__()

        # Three tau "buckets"
        self.fast   = LTCStyleLayer(input_size, hidden_size, tau=1.0,  dt=1.0, learn_tau=False)
        self.medium = LTCStyleLayer(input_size, hidden_size, tau=5.0,  dt=1.0, learn_tau=False)
        self.slow   = LTCStyleLayer(input_size, hidden_size, tau=20.0, dt=1.0, learn_tau=False)

        combined_size = hidden_size * 3

        self.fcn = nn.Sequential(
            nn.Linear(combined_size, fcn_size),
            nn.ReLU(),
            nn.Linear(fcn_size, fcn_size),
            nn.ReLU(),
        )

        # Classification head: single-bounce (logit)
        self.cls_head = nn.Linear(fcn_size, 1)

        # Regression head: distance or distance correction
        self.reg_head = nn.Linear(fcn_size, 1)

    def forward(self, x):
        """
        x: (B, T, 1)  CIR magnitude only
        Returns:
          logit_single  (B, 1)
          prob_single   (B, 1) in [0,1]
          dist_pred     (B, 1)
        """
        _, h_fast   = self.fast(x)    # (B, H)
        _, h_medium = self.medium(x)  # (B, H)
        _, h_slow   = self.slow(x)    # (B, H)

        combined = torch.cat([h_fast, h_medium, h_slow], dim=1)  # (B, 3H)
        feat = self.fcn(combined)                                # (B, fcn_size)

        logit_single = self.cls_head(feat)        # (B, 1)
        prob_single  = torch.sigmoid(logit_single)

        dist_pred = self.reg_head(feat)           # (B, 1)

        return logit_single, prob_single, dist_pred
