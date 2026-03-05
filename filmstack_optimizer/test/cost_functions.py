"""
目标函数（损失）模块：与膜系定义解耦，根据光学计算结果返回标量损失。
支持工业常见目标：最小化/最大化 R 或 T、单波长、波段、匹配目标曲线等。
所有返回的损失均做数值保护（clamp、nan_to_num），避免梯度爆炸导致 NaN。
"""
import torch

# 反射率/透射率在损失中的合理范围，避免除零或矩阵奇异导致无穷
CLAMP_MAX = 2.0


def _safe_R(calc, pol="s"):
    """从 OpticalCalculator 取反射率并夹紧、去 nan/inf。"""
    R = calc.reflection(pol)
    R = torch.clamp(R, 0.0, CLAMP_MAX)
    return torch.nan_to_num(R, nan=0.0, posinf=CLAMP_MAX, neginf=0.0)


def _safe_T(calc, pol="s"):
    """从 OpticalCalculator 取透射率并夹紧、去 nan/inf。"""
    T = calc.transmission(pol)
    T = torch.clamp(T, 0.0, CLAMP_MAX)
    return torch.nan_to_num(T, nan=0.0, posinf=CLAMP_MAX, neginf=0.0)


def cost_minimize_R_s(calc, wavelengths=None, angles=None):
    """
    最小化 s 偏振平均反射率（增透常用）。
    工业惯例：在指定波段内希望 R 越小越好。
    """
    R = _safe_R(calc, "s")
    return R.mean()


def cost_minimize_R_p(calc, wavelengths=None, angles=None):
    """最小化 p 偏振平均反射率。"""
    R = _safe_R(calc, "p")
    return R.mean()


def cost_minimize_R_avg(calc, wavelengths=None, angles=None):
    """最小化 s 与 p 平均反射率的均值（非偏振光近似）。"""
    Rs = _safe_R(calc, "s")
    Rp = _safe_R(calc, "p")
    return (Rs + Rp).mean() / 2.0


def cost_maximize_T_s(calc, wavelengths=None, angles=None):
    """最大化 s 偏振平均透射率（等价于最小化 1-T，常用于滤光片通带）。"""
    T = _safe_T(calc, "s")
    return (1.0 - T).mean()


def cost_maximize_T_avg(calc, wavelengths=None, angles=None):
    """最大化平均透射率（s+p）/2。"""
    Ts = _safe_T(calc, "s")
    Tp = _safe_T(calc, "p")
    T_avg = (Ts + Tp) / 2.0
    return (1.0 - T_avg).mean()


def cost_maximize_R_s(calc, wavelengths=None, angles=None):
    """最大化 s 偏振平均反射率（高反膜、反射镜）。等价于最小化 (1-R)，loss 非负且在 [0,1]。"""
    R = _safe_R(calc, "s")
    return (1.0 - R).mean()


def cost_R_at_wavelength(calc, wavelengths, angles=None, target_wl_nm=550.0, weight_s=1.0, weight_p=0.0):
    """
    最小化在指定波长 target_wl_nm 处的反射率（单波长增透）。
    wavelengths: 计算时用的波长张量，用于找到最接近 target_wl_nm 的索引。
    """
    Rs = _safe_R(calc, "s")
    Rp = _safe_R(calc, "p")
    R = weight_s * Rs + weight_p * Rp
    wl = wavelengths.to(R.device)
    idx = (wl - target_wl_nm).abs().argmin()
    return R.reshape(-1)[idx]


def cost_match_target_R(
    calc, wavelengths, angles=None,
    target_R=None, pol="s",
):
    """
    匹配目标反射率曲线：MSE( R(λ) - target_R(λ) )。
    target_R: 一维张量长度 = 波长数，会按波长维与 R 对齐；None 表示目标为 0（最小化 R）。
    """
    R = _safe_R(calc, pol)  # (L, A)
    if target_R is None:
        return R.mean()
    target_R = target_R.to(R.device).to(R.dtype)
    if target_R.dim() == 0:
        target_R = target_R.expand_as(R)
    else:
        target_R = target_R.reshape(-1, 1).expand(R.shape[0], R.shape[1])
    return ((R - target_R) ** 2).mean()


def cost_match_target_T(
    calc, wavelengths, angles=None,
    target_T=None, pol="s",
):
    """匹配目标透射率曲线：MSE( T(λ) - target_T(λ) )。target_T 形状约定同 cost_match_target_R。"""
    T = _safe_T(calc, pol)
    if target_T is None:
        return (1.0 - T).mean()
    target_T = target_T.to(T.device).to(T.dtype)
    if target_T.dim() == 0:
        target_T = target_T.expand_as(T)
    else:
        target_T = target_T.reshape(-1, 1).expand(T.shape[0], T.shape[1])
    return ((T - target_T) ** 2).mean()


def cost_band_R_max(calc, wavelengths, angles=None, wl_min=450.0, wl_max=650.0):
    """
    最小化指定波段 [wl_min, wl_max] 内的最大反射率（保证波段内 R 不超标）。
    工业上用于“波段内最高反射率”约束。
    """
    R = _safe_R(calc, "s")
    wl = wavelengths.to(R.device)
    mask = (wl >= wl_min) & (wl <= wl_max)
    if mask.any():
        R_band = R.reshape(-1)[mask.reshape(-1)]
        return R_band.max()
    return R.mean()


def cost_band_T_min(calc, wavelengths, angles=None, wl_min=500.0, wl_max=600.0):
    """最大化指定波段内最小透射率（等价于最小化 1 - T_min）。"""
    T = _safe_T(calc, "s")
    wl = wavelengths.to(T.device)
    mask = (wl >= wl_min) & (wl <= wl_max)
    if mask.any():
        T_band = T.reshape(-1)[mask.reshape(-1)]
        return (1.0 - T_band.min())
    return (1.0 - T).mean()


def make_cost_R_at_center(center_nm):
    """工厂：返回一个在 center_nm 处最小化 R 的 cost。"""
    def _cost(calc, wavelengths=None, angles=None):
        return cost_R_at_wavelength(calc, wavelengths, angles, target_wl_nm=center_nm)
    return _cost


def make_cost_match_target_R_curve(target_R_tensor):
    """工厂：返回匹配给定目标反射率曲线的 cost。"""
    def _cost(calc, wavelengths=None, angles=None):
        return cost_match_target_R(calc, wavelengths, angles, target_R=target_R_tensor, pol="s")
    return _cost
