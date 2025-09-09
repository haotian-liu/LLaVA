# Simple projector factory for LLaVA fork
import torch
import torch.nn as nn

__all__ = ["build_projector", "LinearProjector", "MLPProjector"]

class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, T, C_in) or (T, C_in). Keep shape.
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        x = self.drop(self.proj(x))
        if was_2d:
            x = x.squeeze(0)
        return x

class MLPProjector(nn.Module):
    """2-layer MLP with GELU. hidden = hidden_mult * out_dim"""
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, int(hidden_mult * out_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        x = self.net(x)
        if was_2d:
            x = x.squeeze(0)
        return x

def build_projector(kind: str, in_dim: int, out_dim: int, hidden_mult: float = 1.0, dropout: float = 0.0) -> nn.Module:
    kind = (kind or "linear").lower()
    if kind in ("linear", "id", "identity"):
        return LinearProjector(in_dim, out_dim, dropout)
    if kind in ("mlp2x", "mlp", "mlp-gelu"):
        return MLPProjector(in_dim, out_dim, hidden_mult=hidden_mult, dropout=dropout)
    raise ValueError(f"Unknown projector kind: {kind}")
