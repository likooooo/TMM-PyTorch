"""
基于传输矩阵法（TMM）和 PyTorch 自动微分的多层膜优化器。
支持不等式约束：每层厚度 [thickness_min, thickness_max]；可学习 n、k 每变量独立 [n_min,n_max]、[k_min,k_max]。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .path_setup import setup_paths

setup_paths()

from torch_tmm import Layer, Material, OpticalCalculator, T_matrix

from .constraints import (
    thickness_from_raw,
    raw_from_thickness,
    nk_from_raw,
    raw_from_nk,
    THICKNESS_MIN_NM,
    THICKNESS_MAX_NM,
    NK_MIN_DEFAULT,
    NK_MAX_DEFAULT,
)
from .learnable_nk import LearnableNKDispersion

# 梯度裁剪最大范数，防止梯度爆炸导致 NaN
DEFAULT_GRAD_CLIP = 1.0

# 反射率在损失中的合理上界，避免 T[0,0] 接近 0 时 r 爆炸
REFLECTION_CLAMP_MAX = 2.0


def _to_tensor(x, dtype=None, device=None):
    """将 Python 数值或数组转为指定 dtype/device 的 Tensor。"""
    if isinstance(x, (int, float)):
        return torch.tensor(float(x), dtype=dtype or torch.float32, device=device or "cpu")
    return torch.as_tensor(x, dtype=dtype, device=device)


class FilmStackOptimizer(nn.Module):
    """
    多层膜堆栈优化器。
    优化变量：每层的厚度和/或 n、k。
    不等式约束通过重新参数化实现：每层厚度与每变量 n、k 各自独立 [min, max]，用 sigmoid 映射。
    """

    def __init__(
        self,
        wavelengths: torch.Tensor,
        angles: torch.Tensor,
        env_layer: Layer,
        subs_layer: Layer,
        structure_spec: list[dict],
        *,
        thickness_min: float = THICKNESS_MIN_NM,
        thickness_max: float = THICKNESS_MAX_NM,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        """
        structure_spec: 每项为以下之一
          - {"thickness": d, "material": Material}  → 只优化该层厚度
          - {"thickness": d, "n": n0, "k": k0}    → 优化该层厚度与 n、k（常数 n,k 层）
        每层独立范围（未给则用默认）：
          - thickness_min, thickness_max（nm）
          - 可学习 n,k 层：n_min, n_max, k_min, k_max（四个独立，默认 0 / 10）
        wavelengths 单位 nm，angles 单位度。
        """
        super().__init__()
        if device is None:
            device = wavelengths.device if hasattr(wavelengths, "device") else torch.device("cpu")
        self.wavelengths = _to_tensor(wavelengths, dtype=dtype, device=device).reshape(-1)
        self.angles = _to_tensor(angles, dtype=dtype, device=device).reshape(-1)
        self.env_layer = env_layer.to(dtype=dtype, device=device)
        self.subs_layer = subs_layer.to(dtype=dtype, device=device)
        self.structure_spec = structure_spec
        self._dtype = dtype
        self._device = device
        self._c_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self._tm = T_matrix(self._c_dtype, self._device)

        # 每层独立：厚度范围 + 可学习 n,k 时每变量独立范围
        self._thickness_mins = []
        self._thickness_maxs = []
        self._n_mins = []
        self._n_maxs = []
        self._k_mins = []
        self._k_maxs = []
        self._thickness_raws = nn.ParameterList()
        self._n_raws = nn.ParameterList()
        self._k_raws = nn.ParameterList()
        self._fixed_materials = []
        for spec in structure_spec:
            t_min = float(spec.get("thickness_min", thickness_min))
            t_max = float(spec.get("thickness_max", thickness_max))
            self._thickness_mins.append(t_min)
            self._thickness_maxs.append(t_max)
            d0 = spec["thickness"]
            t_raw = raw_from_thickness(
                _to_tensor(d0, dtype=dtype, device=device),
                thickness_min=t_min,
                thickness_max=t_max,
            )
            self._thickness_raws.append(nn.Parameter(t_raw))

            if "material" in spec:
                self._fixed_materials.append(spec["material"].to(dtype=dtype, device=device))
                self._n_raws.append(None)
                self._k_raws.append(None)
                self._n_mins.append(None)
                self._n_maxs.append(None)
                self._k_mins.append(None)
                self._k_maxs.append(None)
            else:
                self._fixed_materials.append(None)
                n_min = float(spec.get("n_min", NK_MIN_DEFAULT))
                n_max = float(spec.get("n_max", NK_MAX_DEFAULT))
                k_min = float(spec.get("k_min", NK_MIN_DEFAULT))
                k_max = float(spec.get("k_max", NK_MAX_DEFAULT))
                self._n_mins.append(n_min)
                self._n_maxs.append(n_max)
                self._k_mins.append(k_min)
                self._k_maxs.append(k_max)
                n0 = spec.get("n", 1.5)
                k0 = spec.get("k", 0.0)
                self._n_raws.append(nn.Parameter(raw_from_nk(
                    _to_tensor(n0, dtype=dtype, device=device), nk_min=n_min, nk_max=n_max
                )))
                self._k_raws.append(nn.Parameter(raw_from_nk(
                    _to_tensor(k0, dtype=dtype, device=device), nk_min=k_min, nk_max=k_max
                )))

    def forward(self, wavelengths: torch.Tensor = None, angles: torch.Tensor = None) -> OpticalCalculator:
        """
        前向计算光学响应。
        不传参时使用 self.wavelengths 和 self.angles。
        返回 OpticalCalculator，可调用 .reflection('s'/'p')、.transmission(...) 等。
        """
        wl = (wavelengths if wavelengths is not None else self.wavelengths).to(self._device)
        ang = (angles if angles is not None else self.angles).to(self._device)
        wl = wl.reshape(-1)
        ang = ang.reshape(-1)

        n_env = self.env_layer.refractive_index(wl)
        n_subs = self.subs_layer.refractive_index(wl)
        n_air = torch.ones_like(n_env, dtype=self._c_dtype, device=self._device)
        th_rad = torch.deg2rad(ang).to(dtype=self._dtype, device=self._device)
        nx = n_env[:, None] * torch.sin(th_rad)[None, :]  # (波长数, 角度数)

        T_s = self._stack_transfer(wl, nx, "s", n_env, n_subs, n_air)
        T_p = self._stack_transfer(wl, nx, "p", n_env, n_subs, n_air)

        return OpticalCalculator(Tm_s=T_s, Tm_p=T_p, n_env=n_env, n_subs=n_subs, nx=nx)

    def _stack_transfer(self, wl, nx, pol, n_env, n_subs, n_air):
        """
        计算单偏振（s 或 p）下的整栈传递矩阵。
        顺序：入射介质 → 空气界面 → 各相干层连乘 → 空气/基底界面。
        """
        L, A = wl.shape[0], nx.shape[1]
        T_tot = torch.eye(2, dtype=self._c_dtype, device=self._device).expand(L, A, 2, 2).clone()

        for i, spec in enumerate(self.structure_spec):
            d = thickness_from_raw(
                self._thickness_raws[i],
                thickness_min=self._thickness_mins[i],
                thickness_max=self._thickness_maxs[i],
            )
            if self._fixed_materials[i] is not None:
                n_l = self._fixed_materials[i].refractive_index(wl)
            else:
                disp = LearnableNKDispersion(
                    self._n_raws[i], self._k_raws[i],
                    n_min=self._n_mins[i], n_max=self._n_maxs[i],
                    k_min=self._k_mins[i], k_max=self._k_maxs[i],
                    dtype=self._dtype, device=self._device,
                )
                n_l = disp.refractive_index(wl)

            T_l = self._tm.coherent_layer(pol=pol, n=n_l, d=d, wavelengths=wl, nx=nx)
            T_tot = T_tot @ T_l

        T_env = (self._tm.interface_s if pol == "s" else self._tm.interface_p)(n_env, n_air, nx)
        T_subs = (self._tm.interface_s if pol == "s" else self._tm.interface_p)(n_air, n_subs, nx)
        return T_env @ T_tot @ T_subs

    def get_thicknesses(self) -> list[torch.Tensor]:
        """当前各层厚度（已落在对应 [thickness_min, thickness_max] 内）。"""
        return [
            thickness_from_raw(
                self._thickness_raws[i],
                thickness_min=self._thickness_mins[i],
                thickness_max=self._thickness_maxs[i],
            )
            for i in range(len(self._thickness_raws))
        ]

    def get_nk(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        """当前各层 (n, k)（在各自 [n_min,n_max]、[k_min,k_max] 内）；固定材料层为 None。"""
        out = []
        for i in range(len(self.structure_spec)):
            if self._fixed_materials[i] is not None:
                out.append(None)
            else:
                n = nk_from_raw(
                    self._n_raws[i],
                    nk_min=self._n_mins[i],
                    nk_max=self._n_maxs[i],
                )
                k = nk_from_raw(
                    self._k_raws[i],
                    nk_min=self._k_mins[i],
                    nk_max=self._k_maxs[i],
                )
                out.append((n, k))
        return out

    def run(
        self,
        steps: int = 200,
        loss_fn=None,
        lr: float = 1e-2,
        optimizer_class=torch.optim.Adam,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        verbose: bool = True,
        min_steps: int = None,
        max_steps: int = None,
        early_stop_patience: int = 25,
        early_stop_tol: float = 1e-6,
    ) -> list[float]:
        """
        执行优化。支持最小/最大迭代次数与基于 loss 变化的提前收敛。

        loss_fn(calc: OpticalCalculator) -> 标量 Tensor。
        grad_clip: 梯度范数裁剪上限，<=0 表示不裁剪。
        min_steps: 至少迭代次数，未达此前不判收敛；默认 None 表示用 min(30, max_steps//2)。
        max_steps: 最多迭代次数；默认 None 表示用 steps。
        early_stop_patience: 判断收敛时查看最近多少步的 loss；默认 25。
        early_stop_tol: 最近 early_stop_patience 步内，相邻 loss 最大变化若 <= 此阈值且已 >= min_steps 则提前退出；默认 1e-6。设为 0 表示禁用提前退出。
        """
        max_steps = max_steps if max_steps is not None else steps
        min_steps = min_steps if min_steps is not None else min(30, max(1, max_steps // 2))
        opt = optimizer_class(self.parameters(), lr=lr)
        history = []

        def default_loss(calc):
            R = calc.reflection("s")
            R = torch.clamp(R, 0.0, REFLECTION_CLAMP_MAX)
            R = torch.nan_to_num(R, nan=0.0, posinf=REFLECTION_CLAMP_MAX, neginf=0.0)
            return R.mean()

        loss_fn = loss_fn or default_loss

        for step in range(max_steps):
            opt.zero_grad()
            calc = self.forward()
            loss = loss_fn(calc)
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f"  step {step + 1}/{max_steps} loss 为 nan/inf，跳过本步")
                history.append(float("nan"))
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
            opt.step()
            history.append(loss.detach().item())

            # 提前收敛：early_stop_tol > 0 时，已 >= min_steps 且最近 early_stop_patience 步内 loss 变化 <= tol 则退出；early_stop_tol <= 0 表示禁用
            if early_stop_tol > 0 and (step + 1) >= min_steps and len(history) >= early_stop_patience:
                recent = history[-early_stop_patience:]
                recent_valid = [x for x in recent if x == x and abs(x) != float("inf")]
                if len(recent_valid) >= 2:
                    max_change = max(
                        abs(recent_valid[i] - recent_valid[i - 1])
                        for i in range(1, len(recent_valid))
                    )
                    if max_change <= early_stop_tol:
                        if verbose:
                            print(f"  early stop at step {step + 1} (loss change <= {early_stop_tol})")
                        break

            if verbose and (step + 1) % 50 == 0:
                print(f"  step {step + 1}/{max_steps} loss = {history[-1]:.6f}")
        return history
