"""
膜系定义模块：与优化逻辑解耦，仅负责构建堆栈（波长、角度、环境、基底、结构）。
符合工业界常见设计场景：单层/多层增透、高反、宽带 AR、指定中心波长等。
每个函数返回一个字典，供 run_optimization 使用。
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPTIMIZER_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_OPTIMIZER_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _OPTIMIZER_ROOT not in sys.path:
    sys.path.insert(0, _OPTIMIZER_ROOT)

from filmstack_optimizer.core import setup_paths
setup_paths()

import torch
from torch_tmm import Material, Layer, Dispersion


def _default_device():
    return torch.device("cpu")


def _default_dtype():
    return torch.float32


def _air_env():
    """入射介质：空气。"""
    return Layer(
        Material([Dispersion.Constant_epsilon(1.0)], name="Air"),
        layer_type="semi-inf",
    )


def _glass_subs(dtype, device, n=1.52):
    """常用基底：光学玻璃 n≈1.52。"""
    mat = Material([Dispersion.Constant_epsilon(n**2)], name="Glass")
    return Layer(mat, layer_type="semi-inf")


def _si_subs(dtype, device):
    """硅基底 n≈3.5（可见区简化）。"""
    mat = Material([Dispersion.Constant_epsilon(3.5**2)], name="Si")
    return Layer(mat, layer_type="semi-inf")


# ---------------------------------------------------------------------------
# 仅优化厚度的膜系（固定材料）
# ---------------------------------------------------------------------------


def stack_single_layer_ar_visible(dtype=None, device=None):
    """
    单层增透（可见光 450–650 nm，正入射）。
    工业惯例：玻璃基底 + 单层 MgF2 或类似低折射率膜（n≈1.38），优化厚度使 R 最小。
    """
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(450, 650, 41, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    # 单层低折射率膜，初始厚度约 100 nm（约 λ/4n @ 550nm 量级）
    layer = Material([Dispersion.Constant_epsilon(1.38**2)], name="MgF2-like")
    structure_spec = [{"thickness": 100.0, "material": layer}]
    return {
        "name": "single_layer_ar_visible",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
    }


def stack_single_layer_ar_center_wavelength(dtype=None, device=None, center_nm=550.0):
    """
    单层增透，针对单一中心波长（如 550 nm）。
    工业上常用于激光腔片或单色光元件，目标在该波长附近 R 最小。
    """
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(center_nm - 50, center_nm + 50, 31, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    layer = Material([Dispersion.Constant_epsilon(1.38**2)], name="Low_n")
    structure_spec = [{"thickness": center_nm / (4 * 1.38), "material": layer}]  # 约 λ/4n 初值
    return {
        "name": "single_layer_ar_center",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
        "center_nm": center_nm,
    }


def stack_two_layer_ar_visible(dtype=None, device=None):
    """
    双层增透（可见光），工业上常见 V 型膜或两层层系。
    高折射率层（如 TiO2 n≈2.3）+ 低折射率层（如 SiO2 n≈1.46），优化两层厚度。
    """
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(450, 650, 51, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    high_n = Material([Dispersion.Constant_epsilon(2.3**2)], name="TiO2-like")
    low_n = Material([Dispersion.Constant_epsilon(1.46**2)], name="SiO2-like")
    structure_spec = [
        {"thickness": 60.0, "material": high_n},
        {"thickness": 95.0, "material": low_n},
    ]
    return {
        "name": "two_layer_ar_visible",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
    }


def stack_three_layer_broadband_ar(dtype=None, device=None):
    """
    三层宽带增透（400–700 nm），工业上用于相机镜头等宽带低反射。
    高-低-高或类似结构，三层厚度均可优化。
    """
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(400, 700, 61, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    high_n = Material([Dispersion.Constant_epsilon(2.1**2)], name="High_n")
    low_n = Material([Dispersion.Constant_epsilon(1.38**2)], name="Low_n")
    structure_spec = [
        {"thickness": 70.0, "material": high_n},
        {"thickness": 100.0, "material": low_n},
        {"thickness": 50.0, "material": high_n},
    ]
    return {
        "name": "three_layer_broadband_ar",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
    }


def stack_single_layer_on_si(dtype=None, device=None):
    """
    单层膜 on 硅基底（可见/近红外），例如 Si 太阳能电池或探测器表面减反。
    优化单层厚度使指定波段反射率最小。
    """
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(500, 700, 41, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _si_subs(dtype, device)
    layer = Material([Dispersion.Constant_epsilon(1.9**2)], name="SiN_like")
    structure_spec = [{"thickness": 80.0, "material": layer}]
    return {
        "name": "single_layer_on_si",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
    }


# ---------------------------------------------------------------------------
# 可优化 n、k 的膜系（无 material，用 n/k 初值）
# ---------------------------------------------------------------------------


def stack_learnable_nk_single_ar(dtype=None, device=None):
    """单层可优化 n、k 的增透（可见光），初值 n=1.5, k=0.01。"""
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(500, 600, 21, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    structure_spec = [{"thickness": 80.0, "n": 1.5, "k": 0.01}]
    return {
        "name": "learnable_nk_single_ar",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
        "nk_max": 10.0,
    }


def stack_learnable_nk_single_high_r(dtype=None, device=None):
    """单层可优化 n、k，目标为高反射（如金属化层初值 n=0.5, k=3）。"""
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(550, 650, 21, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    structure_spec = [{"thickness": 30.0, "n": 0.5, "k": 3.0}]
    return {
        "name": "learnable_nk_single_high_r",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
        "nk_max": 10.0,
    }


def stack_learnable_nk_single_match_target(dtype=None, device=None):
    """单层 n、k 可调，用于匹配目标反射率曲线或单点（由 cost 指定）。"""
    dtype = dtype or _default_dtype()
    device = device or _default_device()
    wl = torch.linspace(520, 580, 31, dtype=dtype, device=device)
    angles = torch.tensor([0.0], dtype=dtype, device=device)
    env = _air_env()
    subs = _glass_subs(dtype, device)
    structure_spec = [{"thickness": 100.0, "n": 1.6, "k": 0.02}]
    return {
        "name": "learnable_nk_match_target",
        "wavelengths": wl,
        "angles": angles,
        "env_layer": env,
        "subs_layer": subs,
        "structure_spec": structure_spec,
        "dtype": dtype,
        "device": device,
        "nk_max": 10.0,
    }


def list_thickness_only_stacks():
    """返回所有仅优化厚度的膜系定义（用于遍历测试）。"""
    return [
        stack_single_layer_ar_visible,
        stack_single_layer_ar_center_wavelength,
        stack_two_layer_ar_visible,
        stack_three_layer_broadband_ar,
        stack_single_layer_on_si,
    ]


def list_learnable_nk_stacks():
    """返回所有可优化 n、k 的膜系定义。"""
    return [
        stack_learnable_nk_single_ar,
        stack_learnable_nk_single_high_r,
        stack_learnable_nk_single_match_target,
    ]
