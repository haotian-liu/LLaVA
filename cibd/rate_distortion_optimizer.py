import torch
import torch.nn as nn
import numpy as np
from collections import deque


class RateDistortionOptimizer:
    """
    率失真优化器
    动态调整β参数以在率失真曲线上找到最优工作点
    """
    def __init__(self, initial_beta=1.0, target_compression=0.5, 
                 adaptation_rate=0.01, window_size=100):
        self.beta = initial_beta
        self.target_compression = target_compression
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.rate_history = deque(maxlen=window_size)
        self.distortion_history = deque(maxlen=window_size)
        self.beta_history = deque(maxlen=window_size)
        
        # Pareto前沿估计
        self.pareto_points = []
        
    def estimate_pareto_slope(self, min_points: int = 5, slope_clip: float = 100.0):
        """
        稳健估计 Pareto 前沿局部斜率 d(distortion)/d(rate)
        
        数值稳定性改进:
        - 过滤 NaN/Inf
        - 检查数据方差避免退化
        - 使用 float64 精度
        - polyfit 失败时回退到差分法
        - 裁剪极端斜率值
        """
        if len(self.rate_history) < min_points:
            return 1.0

        rates = np.asarray(self.rate_history, dtype=np.float64)
        distortions = np.asarray(self.distortion_history, dtype=np.float64)

        # 取最近 K 个点（不强制 10，可随历史长度变化）
        K = min(len(rates), 10)
        r = rates[-K:]
        d = distortions[-K:]

        # 过滤 NaN / Inf
        mask = np.isfinite(r) & np.isfinite(d)
        r, d = r[mask], d[mask]
        if r.size < min_points:
            return 1.0

        # rate 几乎不变 -> 线性回归退化
        if np.allclose(r.var(), 0.0, atol=1e-12):
            return 1.0

        # 尝试 polyfit，失败则回退
        try:
            # 一阶拟合 d ≈ a*r + b
            a = np.polyfit(r, d, 1)[0]
            slope = float(abs(a))
        except Exception:
            # 回退 1：使用末两点差分
            if r.size >= 2:
                dr = r[-1] - r[-2]
                dd = d[-1] - d[-2]
                slope = float(abs(dd / dr)) if abs(dr) > 1e-12 else 1.0
            else:
                # 回退 2：使用中位数差分
                diffr = np.diff(r)
                diffd = np.diff(d)
                valid = np.abs(diffr) > 1e-12
                if valid.any():
                    slope = float(np.median(np.abs(diffd[valid] / diffr[valid])))
                else:
                    slope = 1.0

        # 裁剪，避免异常放大
        slope = float(np.clip(slope, 1e-3, slope_clip))
        return slope
    
    def compute_optimal_beta(self, current_rate, current_distortion):
        """
        根据当前率和失真计算最优β
        使用拉格朗日乘数法
        
        数值稳定性改进:
        - 过滤非法输入 (NaN/Inf)
        - 裁剪 compression_error 避免暴涨
        - 使用 float64 精度
        - 强裁剪 beta 范围
        """
        # 非法数值保护
        if not np.isfinite(current_rate) or not np.isfinite(current_distortion):
            # 直接返回上一次 beta，不更新历史
            return self.beta

        # 记录历史
        self.rate_history.append(float(current_rate))
        self.distortion_history.append(float(current_distortion))
        self.beta_history.append(self.beta)

        # 压缩率误差（裁剪）
        compression_error = float(current_rate - self.target_compression)
        compression_error = float(np.clip(compression_error, -1.0, 1.0))

        # 稳健斜率
        pareto_slope = self.estimate_pareto_slope()

        # 自适应调整
        beta_adjustment = float(self.adaptation_rate * compression_error * pareto_slope)

        # 指数更新（double 精度更稳）
        new_beta = float(self.beta * np.exp(beta_adjustment))

        # 裁剪范围
        self.beta = float(np.clip(new_beta, 0.01, 100.0))
        
        return self.beta
    
    def step(self, task_loss, rd_loss):
        """
        优化步骤
        Args:
            task_loss: 任务损失（如分类损失）
            rd_loss: 率失真损失
        """
        # 分解率失真损失
        current_rate = rd_loss.detach().cpu().item()
        current_distortion = task_loss.detach().cpu().item()
        
        # 计算最优β
        optimal_beta = self.compute_optimal_beta(current_rate, current_distortion)
        
        # 更新Pareto前沿
        self.update_pareto_frontier(current_rate, current_distortion)
        
        return optimal_beta
    
    def update_pareto_frontier(self, rate, distortion):
        """更新Pareto前沿估计"""
        # 添加新点
        new_point = (rate, distortion)
        
        # 过滤被支配的点
        non_dominated = []
        for point in self.pareto_points:
            # 如果新点被支配，不添加
            if point[0] <= rate and point[1] <= distortion:
                return
            # 如果旧点被支配，不保留
            if not (rate <= point[0] and distortion <= point[1]):
                non_dominated.append(point)
                
        non_dominated.append(new_point)
        self.pareto_points = non_dominated
    
    def get_pareto_frontier(self):
        """获取当前Pareto前沿"""
        if not self.pareto_points:
            return [], []
            
        # 按率排序
        sorted_points = sorted(self.pareto_points, key=lambda x: x[0])
        rates = [p[0] for p in sorted_points]
        distortions = [p[1] for p in sorted_points]
        
        return rates, distortions


class AdaptiveRateScheduler:
    """
    自适应压缩率调度器
    在训练过程中逐步增加压缩强度
    """
    def __init__(self, initial_rate=0.1, target_rate=0.5, 
                 warmup_steps=1000, total_steps=10000):
        self.initial_rate = initial_rate
        self.target_rate = target_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_compression_rate(self):
        """获取当前压缩率"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # 预热阶段：线性增加
            progress = self.current_step / self.warmup_steps
            return self.initial_rate + progress * (self.target_rate - self.initial_rate) * 0.5
        else:
            # 主训练阶段：余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            return self.target_rate - cosine_factor * (self.target_rate - self.initial_rate) * 0.5