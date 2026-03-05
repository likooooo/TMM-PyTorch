"""
可学习的 n、k 色散，n 与 k 各有独立的 [min, max] 范围。
与 torch_tmm 的 Material 兼容（提供 epsilon(wavelengths)）。
"""
import torch
from torch import nn

from .constraints import nk_from_raw, NK_MIN_DEFAULT, NK_MAX_DEFAULT


class LearnableNKDispersion(nn.Module):
    """
    色散：epsilon = (n + 1j*k)^2。
    n 约束在 [n_min, n_max]，k 约束在 [k_min, k_max]，四个边界独立。
    """

    def __init__(
        self,
        n_raw: torch.Tensor,
        k_raw: torch.Tensor,
        n_min: float = NK_MIN_DEFAULT,
        n_max: float = NK_MAX_DEFAULT,
        k_min: float = NK_MIN_DEFAULT,
        k_max: float = NK_MAX_DEFAULT,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = n_raw.device
        self._n_min = n_min
        self._n_max = n_max
        self._k_min = k_min
        self._k_max = k_max
        self._dtype = dtype
        self._device = device
        self._n_raw_ref = n_raw
        self._k_raw_ref = k_raw

    def _get_nk(self):
        """从当前 raw 得到约束后的 n、k（各自独立范围）。"""
        n = nk_from_raw(self._n_raw_ref, nk_min=self._n_min, nk_max=self._n_max)
        k = nk_from_raw(self._k_raw_ref, nk_min=self._k_min, nk_max=self._k_max)
        return n, k

    def epsilon(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """复介电常数 (n + 1j*k)^2。"""
        n, k = self._get_nk()
        n = n.to(wavelengths)
        k = k.to(wavelengths)
        nk = n + 1j * k
        return (nk * nk).expand_as(wavelengths)

    def refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """复折射率 n + 1j*k。"""
        n, k = self._get_nk()
        n = n.to(wavelengths)
        k = k.to(wavelengths)
        return (n + 1j * k).expand_as(wavelengths)

    def _prepare_wavelengths(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """与 torch_tmm Material 兼容。"""
        if not torch.is_floating_point(wavelengths):
            raise TypeError("wavelengths must be a floating tensor.")
        if (wavelengths <= 0).any():
            raise ValueError("Wavelengths must be positive.")
        return wavelengths.to(dtype=self._dtype, device=self._device)


def make_learnable_nk_material(
    n_raw: torch.Tensor,
    k_raw: torch.Tensor,
    name: str = "LearnableNK",
    n_min: float = NK_MIN_DEFAULT,
    n_max: float = NK_MAX_DEFAULT,
    k_min: float = NK_MIN_DEFAULT,
    k_max: float = NK_MAX_DEFAULT,
):
    """用给定的 raw 与独立 n/k 范围构造 LearnableNKDispersion 的 Material。"""
    from .path_setup import setup_paths
    setup_paths()
    from torch_tmm import Material
    disp = LearnableNKDispersion(
        n_raw, k_raw,
        n_min=n_min, n_max=n_max, k_min=k_min, k_max=k_max,
    )
    return Material([disp], name=name, requires_grad=True)
