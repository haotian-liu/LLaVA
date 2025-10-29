import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math

class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient flow with outlier clipping
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits, clip_min=None, clip_max=None):
        # Handle num_bits type
        if isinstance(num_bits, torch.Tensor):
            num_bits = int(num_bits.item())
        else:
            num_bits = int(num_bits)
            
        n_levels = 2 ** num_bits
        q_min, q_max = 0, n_levels - 1
        
        # Prevent division by zero
        eps = 1e-8
        scale = scale.clamp(min=eps)
        
        # Apply outlier clipping if provided
        if clip_min is not None and clip_max is not None:
            input = torch.clamp(input, clip_min, clip_max)
        
        # Quantize
        input_scaled = input / scale + zero_point
        input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
        
        # Dequantize
        output = (input_quantized - zero_point) * scale
        
        # Save for backward
        ctx.save_for_backward(input)
        ctx.scale = scale
        ctx.zero_point = zero_point
        ctx.q_min = q_min
        ctx.q_max = q_max
        ctx.clip_min = clip_min
        ctx.clip_max = clip_max
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        # Straight-through gradient
        grad_input = grad_output.clone()
        
        # Zero gradient outside valid range
        input_scaled = input / ctx.scale + ctx.zero_point
        mask = (input_scaled >= ctx.q_min) & (input_scaled <= ctx.q_max)
        
        # Also consider clipping range if applicable
        if ctx.clip_min is not None and ctx.clip_max is not None:
            clip_mask = (input >= ctx.clip_min) & (input <= ctx.clip_max)
            mask = mask & clip_mask
        
        grad_input = grad_input * mask.float()
        
        # Gradients for clip_min and clip_max (if learnable)
        grad_clip_min = grad_clip_max = None
        if ctx.clip_min is not None and ctx.clip_max is not None:
            grad_clip_min = -grad_output[input < ctx.clip_min].sum()
            grad_clip_max = grad_output[input > ctx.clip_max].sum()
        
        return grad_input, None, None, None, grad_clip_min, grad_clip_max


class DifferentiableQuantizer(nn.Module):
    """
    Enhanced differentiable quantizer with multiple granularities and outlier handling
    """
    def __init__(
        self,
        feature_dim: int,
        default_bits: int = 8,
        granularity: Literal['per_tensor', 'per_channel', 'per_group'] = 'per_channel',
        symmetric: bool = False,
        per_sample: bool = False,
        bit_levels: list = [2, 4, 8],
        target_budget: Optional[float] = None,
        # Outlier handling parameters
        use_outlier_clipping: bool = False,
        clip_percentile: float = 99.9,
        learnable_clip: bool = False,
        num_groups: Optional[int] = None  # For per_group granularity
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.default_bits = default_bits
        self.granularity = granularity
        self.symmetric = symmetric
        self.eps = 1e-8
        self.per_sample = per_sample
        self.bit_levels = bit_levels
        self.target_budget = target_budget if target_budget is not None else default_bits
        
        # Outlier handling
        self.use_outlier_clipping = use_outlier_clipping
        self.clip_percentile = clip_percentile
        self.learnable_clip = learnable_clip
        
        # Setup quantization parameters based on granularity
        if granularity == 'per_tensor':
            self.scale = nn.Parameter(torch.ones(1))
            self.zero_point = nn.Parameter(torch.zeros(1))
            param_shape = (1,)
        elif granularity == 'per_channel':
            self.scale = nn.Parameter(torch.ones(1, 1, feature_dim))
            self.zero_point = nn.Parameter(torch.zeros(1, 1, feature_dim))
            param_shape = (1, 1, feature_dim)
        elif granularity == 'per_group':
            if num_groups is None:
                num_groups = max(4, feature_dim // 128)  # Default grouping
            self.num_groups = num_groups
            self.group_size = feature_dim // num_groups
            # Parameters per group
            self.scale = nn.Parameter(torch.ones(1, 1, num_groups))
            self.zero_point = nn.Parameter(torch.zeros(1, 1, num_groups))
            param_shape = (1, 1, num_groups)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        # Calibration stats
        self.register_buffer('running_min', torch.zeros(param_shape))
        self.register_buffer('running_max', torch.zeros(param_shape))
        self.register_buffer('calibration_counter', torch.tensor(0))
        
        # Outlier clipping parameters
        if use_outlier_clipping:
            if learnable_clip:
                self.clip_min = nn.Parameter(torch.zeros(param_shape))
                self.clip_max = nn.Parameter(torch.ones(param_shape))
            else:
                self.register_buffer('clip_min', torch.zeros(param_shape))
                self.register_buffer('clip_max', torch.ones(param_shape))
        else:
            self.clip_min = None
            self.clip_max = None
    
    def _reshape_for_granularity(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input based on quantization granularity"""
        batch_size, seq_len, feature_dim = x.shape
        
        if self.granularity == 'per_group':
            # Reshape to [B, T, num_groups, group_size]
            x = x.view(batch_size, seq_len, self.num_groups, self.group_size)
            # Transpose to [B, T, group_size, num_groups] for easier processing
            x = x.transpose(-1, -2).contiguous()
            # Flatten to [B*T*group_size, num_groups]
            x = x.view(-1, self.num_groups)
        
        return x
    
    def _restore_shape(self, x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Restore original shape after quantization"""
        if self.granularity == 'per_group':
            batch_size, seq_len, feature_dim = original_shape
            # Reshape back from [B*T*group_size, num_groups]
            x = x.view(batch_size, seq_len, self.group_size, self.num_groups)
            # Transpose back to [B, T, num_groups, group_size]
            x = x.transpose(-1, -2).contiguous()
            # Flatten last two dimensions to [B, T, feature_dim]
            x = x.view(batch_size, seq_len, feature_dim)
        
        return x
    
    def _compute_outlier_clips(self, x: torch.Tensor):
        """Compute outlier clipping thresholds"""
        with torch.no_grad():
            if self.granularity == 'per_tensor':
                flat_x = x.flatten()
                clip_min = torch.quantile(flat_x, (100 - self.clip_percentile) / 100)
                clip_max = torch.quantile(flat_x, self.clip_percentile / 100)
            elif self.granularity == 'per_channel':
                # Compute per-channel percentiles
                flat_x = x.view(-1, x.size(-1))  # [B*T, D]
                clip_min = torch.quantile(flat_x, (100 - self.clip_percentile) / 100, dim=0, keepdim=True)
                clip_max = torch.quantile(flat_x, self.clip_percentile / 100, dim=0, keepdim=True)
                clip_min = clip_min.unsqueeze(0)  # [1, 1, D]
                clip_max = clip_max.unsqueeze(0)
            elif self.granularity == 'per_group':
                # Reshape for groups
                x_grouped = self._reshape_for_granularity(x)
                clip_min = torch.quantile(x_grouped, (100 - self.clip_percentile) / 100, dim=0, keepdim=True)
                clip_max = torch.quantile(x_grouped, self.clip_percentile / 100, dim=0, keepdim=True)
                clip_min = clip_min.unsqueeze(0)  # [1, 1, num_groups]
                clip_max = clip_max.unsqueeze(0)
            
            if not self.learnable_clip:
                self.clip_min.copy_(clip_min)
                self.clip_max.copy_(clip_max)
            else:
                # Initialize learnable parameters
                self.clip_min.data.copy_(clip_min)
                self.clip_max.data.copy_(clip_max)
    
    def forward(
        self,
        x: torch.Tensor,
        bit_assignment: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        original_shape = x.shape
        
        # Calibrate if needed
        if self.training and self.calibration_counter < 100:
            self.calibrate(x)
        
        # Handle outlier clipping initialization
        if self.use_outlier_clipping and self.calibration_counter == 100:
            self._compute_outlier_clips(x)
        
        # Prepare clipping values based on granularity
        clip_min = clip_max = None
        if self.use_outlier_clipping and self.clip_min is not None:
            if self.granularity == 'per_group':
                # Expand clips for group-wise application
                clip_min = self.clip_min.expand(-1, -1, self.num_groups)
                clip_max = self.clip_max.expand(-1, -1, self.num_groups)
            else:
                clip_min = self.clip_min
                clip_max = self.clip_max
        
        if bit_assignment is not None and group_indices is not None:
            # Discretize bit assignment with budget constraint
            discrete_bits = self.discretize_with_budget(
                bit_assignment, 
                self.target_budget, 
                self.bit_levels
            )
            
            # Apply adaptive quantization
            if self.per_sample:
                return self.adaptive_quantize_per_sample(
                    x, discrete_bits, group_indices, clip_min, clip_max
                )
            else:
                return self.adaptive_quantize(
                    x, discrete_bits, group_indices, clip_min, clip_max
                )
        else:
            # Uniform quantization with granularity support
            if self.granularity == 'per_group':
                x = self._reshape_for_granularity(x)
            
            if self.training:
                x_quantized = StraightThroughEstimator.apply(
                    x, self.scale, self.zero_point, self.default_bits,
                    clip_min, clip_max
                )
            else:
                # Apply clipping if enabled
                if clip_min is not None and clip_max is not None:
                    x = torch.clamp(x, clip_min, clip_max)
                
                n_levels = 2 ** self.default_bits
                q_min, q_max = 0, n_levels - 1
                scale_safe = self.scale.clamp(min=self.eps)
                input_scaled = x / scale_safe + self.zero_point
                input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                x_quantized = (input_quantized - self.zero_point) * scale_safe
            
            if self.granularity == 'per_group':
                x_quantized = self._restore_shape(x_quantized, original_shape)
            
            return x_quantized
    
    def calibrate(self, x: torch.Tensor):
        """Update calibration statistics"""
        with torch.no_grad():
            if self.granularity == 'per_tensor':
                x_min = x.min()
                x_max = x.max()
            elif self.granularity == 'per_channel':
                x_min = x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            elif self.granularity == 'per_group':
                x_grouped = self._reshape_for_granularity(x)
                x_min = x_grouped.min(dim=0, keepdim=True)[0]
                x_max = x_grouped.max(dim=0, keepdim=True)[0]
                x_min = x_min.unsqueeze(0)
                x_max = x_max.unsqueeze(0)
            
            if self.calibration_counter == 0:
                self.running_min.copy_(x_min)
                self.running_max.copy_(x_max)
            else:
                momentum = 0.9
                self.running_min.mul_(momentum).add_(x_min, alpha=1-momentum)
                self.running_max.mul_(momentum).add_(x_max, alpha=1-momentum)
            
            self.calibration_counter += 1
            
            # Update scale and zero_point
            if self.symmetric:
                abs_max = torch.max(self.running_max.abs(), self.running_min.abs())
                abs_max = abs_max.clamp(min=self.eps)
                self.scale.data = abs_max / (2 ** (self.default_bits - 1) - 1)
                self.scale.data.clamp_(min=self.eps)
                self.zero_point.data.zero_()
            else:
                range_val = self.running_max - self.running_min
                range_val = range_val.clamp(min=self.eps)
                self.scale.data = range_val / (2 ** self.default_bits - 1)
                self.scale.data.clamp_(min=self.eps)
                self.zero_point.data = -self.running_min / self.scale.data
    
    def adaptive_quantize(
        self,
        x: torch.Tensor,
        bit_assignment: torch.Tensor,
        group_indices: torch.Tensor,
        clip_min: Optional[torch.Tensor] = None,
        clip_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply mixed-precision quantization (batch-level)"""
        batch_size, seq_len, feature_dim = x.shape
        num_groups = bit_assignment.size(1)
        x_quantized = torch.zeros_like(x)
        
        for g in range(num_groups):
            # Get group mask and features
            group_mask = (group_indices == g).unsqueeze(1).expand_as(x)
            group_features = x * group_mask.float()
            
            # Get bits for this group
            group_bits = int(bit_assignment[:, g].mean().item())
            
            # Apply group-specific clipping if available
            group_clip_min = group_clip_max = None
            if clip_min is not None and self.granularity == 'per_group':
                group_clip_min = clip_min[:, :, g:g+1]
                group_clip_max = clip_max[:, :, g:g+1]
            elif clip_min is not None:
                group_clip_min = clip_min
                group_clip_max = clip_max
            
            # Quantize group
            if self.training:
                group_quantized = StraightThroughEstimator.apply(
                    group_features,
                    self.scale,
                    self.zero_point,
                    group_bits,
                    group_clip_min,
                    group_clip_max
                )
            else:
                if group_clip_min is not None:
                    group_features = torch.clamp(group_features, group_clip_min, group_clip_max)
                
                n_levels = 2 ** group_bits
                q_min, q_max = 0, n_levels - 1
                scale_safe = self.scale.clamp(min=self.eps)
                input_scaled = group_features / scale_safe + self.zero_point
                input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                group_quantized = (input_quantized - self.zero_point) * scale_safe
            
            x_quantized = x_quantized + group_quantized * group_mask.float()
        
        return x_quantized
    
    def adaptive_quantize_per_sample(
        self,
        x: torch.Tensor,
        bit_assignment: torch.Tensor,
        group_indices: torch.Tensor,
        clip_min: Optional[torch.Tensor] = None,
        clip_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply mixed-precision quantization (per-sample level)"""
        batch_size, seq_len, feature_dim = x.shape
        num_groups = bit_assignment.size(1)
        x_quantized = torch.zeros_like(x)
        
        for b in range(batch_size):
            for g in range(num_groups):
                # Get group mask for this sample
                if group_indices.dim() == 2:  # [B, D]
                    group_mask = (group_indices[b] == g).unsqueeze(0).expand(seq_len, -1)
                else:  # [B, T, D]
                    group_mask = (group_indices[b] == g)
                
                # Get features for this group and sample
                sample_features = x[b] * group_mask.float()
                
                # Get bits for this specific sample and group
                sample_bits = int(bit_assignment[b, g].item())
                
                # Prepare clipping for this group if applicable
                sample_clip_min = sample_clip_max = None
                if clip_min is not None:
                    if self.granularity == 'per_group':
                        sample_clip_min = clip_min[:, :, g:g+1]
                        sample_clip_max = clip_max[:, :, g:g+1]
                    else:
                        sample_clip_min = clip_min
                        sample_clip_max = clip_max
                
                # Quantize
                if self.training:
                    sample_features_expanded = sample_features.unsqueeze(0)
                    
                    # Get appropriate scale/zero_point slice
                    if self.granularity == 'per_channel':
                        group_channels = group_mask[0] if seq_len > 0 else group_mask
                        scale_group = self.scale[:, :, group_channels]
                        zero_point_group = self.zero_point[:, :, group_channels]
                    elif self.granularity == 'per_group':
                        scale_group = self.scale[:, :, g:g+1]
                        zero_point_group = self.zero_point[:, :, g:g+1]
                    else:
                        scale_group = self.scale
                        zero_point_group = self.zero_point
                    
                    group_quantized = StraightThroughEstimator.apply(
                        sample_features_expanded,
                        scale_group,
                        zero_point_group,
                        sample_bits,
                        sample_clip_min,
                        sample_clip_max
                    )
                    x_quantized[b] += group_quantized.squeeze(0) * group_mask.float()
                else:
                    # Apply clipping if enabled
                    if sample_clip_min is not None:
                        sample_features = torch.clamp(sample_features, sample_clip_min, sample_clip_max)
                    
                    # Inference mode quantization
                    n_levels = 2 ** sample_bits
                    q_min, q_max = 0, n_levels - 1
                    
                    if self.granularity == 'per_channel':
                        group_channels = group_mask[0] if seq_len > 0 else group_mask
                        scale_safe = self.scale[:, :, group_channels].squeeze(0).clamp(min=self.eps)
                        zero_point_use = self.zero_point[:, :, group_channels].squeeze(0)
                    elif self.granularity == 'per_group':
                        scale_safe = self.scale[:, :, g:g+1].squeeze().clamp(min=self.eps)
                        zero_point_use = self.zero_point[:, :, g:g+1].squeeze()
                    else:
                        scale_safe = self.scale.clamp(min=self.eps)
                        zero_point_use = self.zero_point
                    
                    input_scaled = sample_features / scale_safe + zero_point_use
                    input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                    group_quantized = (input_quantized - zero_point_use) * scale_safe
                    
                    x_quantized[b] += group_quantized * group_mask.float()
        
        return x_quantized
    
    def discretize_with_budget(
        self, 
        bit_assignment: torch.Tensor,
        target_budget: float,
        bit_levels: list
    ) -> torch.Tensor:
        """
        Discretize bit assignment to allowed levels and enforce budget constraint
        """
        batch_size, num_groups = bit_assignment.shape
        bit_levels_tensor = torch.tensor(bit_levels, device=bit_assignment.device, dtype=torch.float32)
        
        # Step 1: Find nearest discrete level for each assignment
        distances = torch.abs(bit_assignment.unsqueeze(-1) - bit_levels_tensor.view(1, 1, -1))
        discrete_bits = bit_levels_tensor[distances.argmin(dim=-1)]
        
        # Step 2: Adjust to meet budget constraint per sample
        for b in range(batch_size):
            current_budget = discrete_bits[b].mean()
            iterations = 0
            max_iterations = num_groups * 2
            
            while abs(current_budget - target_budget) > 0.1 and iterations < max_iterations:
                iterations += 1
                
                if current_budget > target_budget:
                    # Need to reduce bits
                    reducible = discrete_bits[b] > min(bit_levels)
                    if not reducible.any():
                        break
                    
                    reducible_indices = torch.where(reducible)[0]
                    max_bit_idx = reducible_indices[discrete_bits[b, reducible_indices].argmax()]
                    
                    current_level_idx = (discrete_bits[b, max_bit_idx] == bit_levels_tensor).nonzero()[0, 0]
                    if current_level_idx > 0:
                        discrete_bits[b, max_bit_idx] = bit_levels_tensor[current_level_idx - 1]
                else:
                    # Need to increase bits
                    increasable = discrete_bits[b] < max(bit_levels)
                    if not increasable.any():
                        break
                    
                    increasable_indices = torch.where(increasable)[0]
                    min_bit_idx = increasable_indices[discrete_bits[b, increasable_indices].argmin()]
                    
                    current_level_idx = (discrete_bits[b, min_bit_idx] == bit_levels_tensor).nonzero()[0, 0]
                    if current_level_idx < len(bit_levels) - 1:
                        discrete_bits[b, min_bit_idx] = bit_levels_tensor[current_level_idx + 1]
                
                current_budget = discrete_bits[b].mean()
        
        return discrete_bits
    
    def get_quantization_stats(self) -> dict:
        """Get current quantization statistics"""
        stats = {
            'scale': self.scale.data.clone(),
            'zero_point': self.zero_point.data.clone(),
            'running_min': self.running_min.clone(),
            'running_max': self.running_max.clone(),
            'calibration_counter': self.calibration_counter.item(),
            'granularity': self.granularity
        }
        
        if self.use_outlier_clipping and self.clip_min is not None:
            stats['clip_min'] = self.clip_min.data.clone() if isinstance(self.clip_min, nn.Parameter) else self.clip_min.clone()
            stats['clip_max'] = self.clip_max.data.clone() if isinstance(self.clip_max, nn.Parameter) else self.clip_max.clone()
        
        return stats