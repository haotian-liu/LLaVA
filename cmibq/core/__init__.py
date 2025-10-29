from .ib_quantizer import IBQuantizedLayer
from .importance_estimator import HybridImportanceEstimation, UniversalImportanceEstimation
from .bit_allocator import BitAllocationNetwork
from .differentiable_quant import DifferentiableQuantizer, StraightThroughEstimator

__all__ = [
    'IBQuantizedLayer',
    'HybridImportanceEstimation',  # ← 改正这里
    'UniversalImportanceEstimation',  # ← 添加这个
    'BitAllocationNetwork', 
    'DifferentiableQuantizer',
    'StraightThroughEstimator'
]