"""Cross-Modality Information Bottleneck Quantization (CM-IBQ).

This module implements the CM-IBQ framework described in the paper-like
specification provided by the user prompt.  It exposes three key building
blocks (importance estimation, information-guided bit allocation and a
straight-through differentiable quantizer) as well as a thin wrapper module
(`IBQuantizedLayer`) that can be dropped into an existing vision-language
model (VLM) as the modality fusion bottleneck.

Two high level helper routines – :func:`stage_one_pretrain` and
:func:`stage_two_finetune` – realise the progressive two-stage optimisation
procedure discussed in the specification.  They are intentionally written in a
framework agnostic fashion so that they can be re-used across different LLaVA
setups or even ported to other VLM implementations with minimal changes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class CMIBQConfig:
    """Configuration container for CM-IBQ components.

    Attributes
    ----------
    feature_dim: int
        Dimensionality of the fused multimodal representation that is going to
        be quantised.
    task_dim: int
        Dimensionality of the task embedding fed into the importance estimator.
    hidden_dim: int
        Width of the shared hidden representation used inside the auxiliary
        networks.
    latent_dim: Optional[int]
        Size of the latent bottleneck.  If ``None`` (the default) the latent
        dimension equals ``feature_dim``.
    min_bits: int
        Lower bound on the number of bits that the bit allocation network may
        assign to an individual feature channel.
    max_bits: int
        Upper bound on the number of bits assignable to a channel.
    stage1_bits: int
        Target average bit-width used during the bottleneck shaping pretraining
        phase.
    stage2_bits: int
        Final target bit-width used during task-aware finetuning.
    kl_weight: float
        Weight for the KL divergence term in the IB loss.
    bitrate_weight: float
        Weight for the bitrate regularisation term in stage two.
    alignment_weight: float
        Weight for the cross-modal alignment objective.
    alignment_temperature: float
        Temperature used in the InfoNCE-style alignment loss.
    progressive_anneal_steps: int
        Number of update steps over which the bit-width is annealed from
        ``stage1_bits`` to ``stage2_bits``.
    """

    feature_dim: int
    task_dim: int
    hidden_dim: int = 1024
    latent_dim: Optional[int] = None
    min_bits: int = 2
    max_bits: int = 8
    stage1_bits: int = 8
    stage2_bits: int = 4
    kl_weight: float = 1e-3
    bitrate_weight: float = 1.0
    alignment_weight: float = 0.2
    alignment_temperature: float = 0.07
    progressive_anneal_steps: int = 2000


class ImportanceEstimationNetwork(nn.Module):
    """Task-aware importance estimator.

    The network receives latent features and a task embedding, concatenates the
    two and predicts a per-dimension importance score.  Higher scores indicate
    that a given channel is more relevant for downstream task performance.
    """

    def __init__(self, feature_dim: int, task_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + task_dim),
            nn.Linear(feature_dim + task_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, latent: Tensor, task_embedding: Tensor) -> Tensor:
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).expand_as(latent)
        elif task_embedding.dim() == 2 and task_embedding.size(0) == 1:
            task_embedding = task_embedding.expand(latent.size(0), -1)
        concatenated = torch.cat([latent, task_embedding], dim=-1)
        return self.net(concatenated)


class BitAllocationNetwork(nn.Module):
    """Learns to allocate bits according to importance scores.

    The network maps importance scores to soft bit-widths, which are later
    rounded when executing quantisation.  A differentiable normalisation step
    ensures that the allocated bits respect a desired average bit budget.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        min_bits: int = 2,
        max_bits: int = 8,
    ) -> None:
        super().__init__()
        self.min_bits = float(min_bits)
        self.max_bits = float(max_bits)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, importance_scores: Tensor, target_average_bits: float) -> Tensor:
        raw = self.net(importance_scores)
        soft_alloc = torch.sigmoid(raw)
        bits = self.min_bits + (self.max_bits - self.min_bits) * soft_alloc
        mean_bits = bits.mean(dim=-1, keepdim=True) + 1e-6
        scaling = target_average_bits / mean_bits
        scaled_bits = torch.clamp(bits * scaling, self.min_bits, self.max_bits)
        return scaled_bits


class _STEQuantizeFn(torch.autograd.Function):
    """Straight-through estimator for the quantise/dequantise round-trip."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, zero_point: Tensor, qmin: Tensor, qmax: Tensor) -> Tensor:
        ctx.save_for_backward(x, scale, zero_point, qmin, qmax)
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        x, scale, _, qmin, qmax = ctx.saved_tensors
        mask = (x >= (qmin - 0.5) * scale) & (x <= (qmax + 0.5) * scale)
        return grad_output * mask, None, None, None, None


class DifferentiableQuantizer(nn.Module):
    """Adaptive differentiable quantiser based on min-max calibration."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _compute_scale_zero_point(x: Tensor, bits: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        qmax = torch.pow(2.0, bits) - 1.0
        qmin = torch.zeros_like(qmax)
        x_min = x.amin(dim=0, keepdim=True)
        x_max = x.amax(dim=0, keepdim=True)
        scale = (x_max - x_min).clamp(min=eps) / torch.clamp(qmax, min=1.0)
        zero_point = torch.clamp(torch.round(-x_min / scale), qmin, qmax)
        scale = scale.expand_as(x)
        zero_point = zero_point.expand_as(x)
        qmin = qmin.expand_as(x)
        qmax = qmax.expand_as(x)
        return scale, zero_point, qmin, qmax

    def forward(self, x: Tensor, bits: Tensor) -> Tensor:
        scale, zero_point, qmin, qmax = self._compute_scale_zero_point(x, bits, self.eps)
        return _STEQuantizeFn.apply(x, scale, zero_point, qmin, qmax)


@dataclass
class IBLossComponents:
    reconstruction: Tensor
    kl: Tensor
    bitrate: Optional[Tensor] = None
    alignment: Optional[Tensor] = None

    def total(self, kl_weight: float, bitrate_weight: float, alignment_weight: float) -> Tensor:
        loss = self.reconstruction + kl_weight * self.kl
        if self.bitrate is not None:
            loss = loss + bitrate_weight * self.bitrate
        if self.alignment is not None:
            loss = loss + alignment_weight * self.alignment
        return loss


class IBQuantizedLayer(nn.Module):
    """End-to-end learnable information bottleneck with adaptive quantisation."""

    def __init__(self, config: CMIBQConfig) -> None:
        super().__init__()
        latent_dim = config.latent_dim or config.feature_dim
        self.config = config
        self.encoder = nn.Linear(config.feature_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, config.feature_dim)
        self.importance = ImportanceEstimationNetwork(latent_dim, config.task_dim, config.hidden_dim)
        self.bit_allocator = BitAllocationNetwork(latent_dim, config.hidden_dim // 2, config.min_bits, config.max_bits)
        self.quantizer = DifferentiableQuantizer()
        self.latent_dim = latent_dim
        self.register_buffer("running_target_bits", torch.tensor(float(config.stage1_bits)))

    def reparameterise(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def anneal_bits(self, step: int) -> float:
        if self.config.progressive_anneal_steps <= 0:
            return float(self.config.stage2_bits)
        progress = min(step / self.config.progressive_anneal_steps, 1.0)
        current = self.config.stage1_bits + progress * (self.config.stage2_bits - self.config.stage1_bits)
        self.running_target_bits.fill_(current)
        return float(current)

    def forward(
        self,
        features: Tensor,
        task_embedding: Tensor,
        *,
        target_bits: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        latent_params = self.encoder(features)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        latent = self.reparameterise(mu, logvar)
        importance_scores = self.importance(latent, task_embedding)
        bit_allocation = self.bit_allocator(importance_scores, target_bits or float(self.running_target_bits.item()))
        discrete_bits = torch.clamp(torch.round(bit_allocation), self.config.min_bits, self.config.max_bits)
        soft_bits = bit_allocation + (discrete_bits - bit_allocation).detach()
        quantised_latent = self.quantizer(latent, soft_bits)
        reconstructed = self.decoder(quantised_latent)
        return quantised_latent, reconstructed, mu, logvar, soft_bits

    def compute_losses(
        self,
        features: Tensor,
        reconstructed: Tensor,
        mu: Tensor,
        logvar: Tensor,
        bit_allocation: Tensor,
        *,
        target_bits: float,
        text_features: Optional[Tensor] = None,
        quantised_latent: Optional[Tensor] = None,
    ) -> IBLossComponents:
        reconstruction_loss = F.mse_loss(reconstructed, features)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        bitrate_loss = (bit_allocation.mean(dim=-1) - target_bits).abs().mean()
        alignment_loss = None
        if text_features is not None:
            anchor = quantised_latent if quantised_latent is not None else reconstructed
            alignment_loss = self.contrastive_alignment_loss(anchor, text_features)
        return IBLossComponents(
            reconstruction=reconstruction_loss,
            kl=kl_loss,
            bitrate=bitrate_loss,
            alignment=alignment_loss,
        )

    def contrastive_alignment_loss(self, vision_features: Tensor, text_features: Tensor) -> Tensor:
        vision = F.normalize(vision_features, dim=-1)
        text = F.normalize(text_features, dim=-1)
        logits = vision @ text.t() / self.config.alignment_temperature
        labels = torch.arange(vision.size(0), device=vision.device)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_v2t + loss_t2v)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(trainable)


def stage_one_pretrain(
    layer: IBQuantizedLayer,
    dataloader: Iterable,
    optimiser: torch.optim.Optimizer,
    *,
    device: torch.device,
    gamma: float = 1.0,
    max_steps: Optional[int] = None,
) -> None:
    """Stage one: bottleneck shaping & importance pre-training.

    Only the quantisation layer parameters are updated.  The surrounding VLM
    should remain frozen when calling this routine.
    """

    layer.train()
    _set_trainable(layer, True)
    step = 0
    for batch in dataloader:
        if max_steps is not None and step >= max_steps:
            break
        features, task_embedding = batch["features"].to(device), batch["task_embedding"].to(device)
        optimiser.zero_grad()
        quantised, reconstructed, mu, logvar, bit_allocation = layer(
            features,
            task_embedding,
            target_bits=layer.config.stage1_bits,
        )
        losses = layer.compute_losses(
            features,
            reconstructed,
            mu,
            logvar,
            bit_allocation,
            target_bits=float(layer.config.stage1_bits),
            quantised_latent=quantised,
        )
        total_loss = losses.reconstruction + gamma * losses.kl
        total_loss.backward()
        optimiser.step()
        step += 1


def stage_two_finetune(
    layer: IBQuantizedLayer,
    model: nn.Module,
    dataloader: Iterable,
    optimiser: torch.optim.Optimizer,
    *,
    device: torch.device,
    anneal: bool = True,
    max_steps: Optional[int] = None,
) -> None:
    """Stage two: joint optimisation with the host model.

    ``model`` is expected to make use of ``layer`` inside its forward pass.  The
    dataloader has to yield dictionaries containing at least ``vision`` and
    ``text`` tensors as well as task labels used by the model's task loss.
    """

    layer.train()
    model.train()
    step = 0
    for batch in dataloader:
        if max_steps is not None and step >= max_steps:
            break
        optimiser.zero_grad()
        if anneal:
            target_bits = layer.anneal_bits(step)
        else:
            target_bits = float(layer.config.stage2_bits)
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        outputs = model(**batch, target_bits=target_bits)
        task_loss: Tensor = outputs["loss"]
        aux = outputs.get("cm_ibq")
        if aux is None:
            raise RuntimeError("Model must return CM-IBQ diagnostics under the 'cm_ibq' key during stage two training.")
        losses = layer.compute_losses(
            aux["features"],
            aux["reconstructed"],
            aux["mu"],
            aux["logvar"],
            aux["bit_allocation"],
            target_bits=target_bits,
            text_features=aux.get("text_features"),
            quantised_latent=aux.get("quantised"),
        )
        total_loss = task_loss
        total_loss = total_loss + layer.config.kl_weight * losses.kl
        bitrate = losses.bitrate if losses.bitrate is not None else torch.tensor(0.0, device=device)
        total_loss = total_loss + layer.config.bitrate_weight * bitrate
        if losses.alignment is not None:
            total_loss = total_loss + layer.config.alignment_weight * losses.alignment
        total_loss.backward()
        optimiser.step()
        step += 1
