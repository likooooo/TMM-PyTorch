"""
不等式约束的重新参数化：每个优化变量都有独立的 [min, max] 范围。
厚度：每层 thickness_min, thickness_max；
n、k：每层（或每个变量）n_min, n_max, k_min, k_max 独立。
"""
import torch

# 厚度默认范围（nm）：最小值为参考波长的 1/4（λ/4 @ 550 nm）
_REFERENCE_WAVELENGTH_NM = 550.0
THICKNESS_MIN_NM = _REFERENCE_WAVELENGTH_NM / 4.0  # ≈ 137.5 nm
THICKNESS_MAX_NM = 5000.0

# n、k 默认范围（单变量未指定时用）
NK_MIN_DEFAULT = 0.0
NK_MAX_DEFAULT = 10.0


def thickness_from_raw(
    raw: torch.Tensor,
    thickness_min: float = THICKNESS_MIN_NM,
    thickness_max: float = THICKNESS_MAX_NM,
) -> torch.Tensor:
    """
    将无约束标量 raw 映射为厚度 in [thickness_min, thickness_max]。
    thickness = thickness_min + (thickness_max - thickness_min) * sigmoid(raw)
    """
    span = thickness_max - thickness_min
    t = thickness_min + span * torch.sigmoid(raw)
    return torch.clamp(t, min=thickness_min, max=thickness_max)


def nk_from_raw(
    raw: torch.Tensor,
    nk_min: float = NK_MIN_DEFAULT,
    nk_max: float = NK_MAX_DEFAULT,
) -> torch.Tensor:
    """
    将无约束标量映射到 [nk_min, nk_max]。
    value = nk_min + (nk_max - nk_min) * sigmoid(raw)，n 与 k 各自独立范围。
    """
    span = nk_max - nk_min
    return nk_min + span * torch.sigmoid(raw)


def raw_from_thickness(
    thickness: torch.Tensor,
    thickness_min: float = THICKNESS_MIN_NM,
    thickness_max: float = THICKNESS_MAX_NM,
) -> torch.Tensor:
    """thickness_from_raw 的反函数，用于用目标厚度初始化 raw。"""
    span = thickness_max - thickness_min
    if span <= 0:
        span = 1e-6
    t = thickness.clamp(min=thickness_min, max=thickness_max)
    x = ((t - thickness_min) / span).clamp(1e-7, 1.0 - 1e-7)
    return torch.log(x / (1.0 - x))


def raw_from_nk(
    value: torch.Tensor,
    nk_min: float = NK_MIN_DEFAULT,
    nk_max: float = NK_MAX_DEFAULT,
) -> torch.Tensor:
    """nk_from_raw 的反函数：value in [nk_min, nk_max] -> raw。"""
    span = nk_max - nk_min
    if span <= 0:
        span = 1e-6
    x = ((value - nk_min) / span).clamp(1e-7, 1.0 - 1e-7)
    return torch.log(x / (1.0 - x))
