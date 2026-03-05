"""
Load n,k from dep/refractiveindex.py (Material) and convert to torch tensors for use in TMM.
Compatible with dep's Material: getRefractiveIndex(wavelength_nm), getExtinctionCoefficient(wavelength_nm).
"""
import os
import numpy as np
import torch

from .path_setup import setup_paths, get_dep_path


def _ensure_dep_in_path():
    # refractiveindex.py lives in filmstack_optimizer/dep/
    dep = get_dep_path()
    sys = __import__("sys")
    if dep not in sys.path:
        sys.path.insert(0, dep)


def load_nk_torch(wavelengths_nm: torch.Tensor, material) -> torch.Tensor:
    """
    Evaluate complex refractive index n + 1j*k at wavelengths_nm using dep Material.
    material must have getRefractiveIndex(wl_nm) and getExtinctionCoefficient(wl_nm)
    (or only getRefractiveIndex for non-absorbing). wavelengths_nm in nm.
    Returns complex tensor same shape as wavelengths_nm.
    """
    wl = wavelengths_nm.detach().cpu().numpy()
    if np.isscalar(wl):
        wl = np.array([wl])
    try:
        n = material.getRefractiveIndex(wl)
    except Exception:
        n = np.ones_like(wl)
    try:
        k = material.getExtinctionCoefficient(wl)
    except Exception:
        k = np.zeros_like(wl)
    if np.isscalar(n):
        n = np.full_like(wl, n)
    if np.isscalar(k):
        k = np.full_like(wl, k)
    nk = (np.asarray(n, dtype=np.complex128) + 1j * np.asarray(k, dtype=np.complex128))
    return torch.from_numpy(nk).to(dtype=wavelengths_nm.dtype, device=wavelengths_nm.device)


def material_from_dep(book: str, page=None, shelf="main", database_path=None):
    """
    Load dep RefractiveIndex and return Material(book=..., page=..., shelf=...).
    Optionally database_path for RefractiveIndex(databasePath=...).
    """
    _ensure_dep_in_path()
    from refractiveindex import RefractiveIndex

    kwargs = {}
    if database_path is not None:
        kwargs["databasePath"] = database_path
    ri = RefractiveIndex(**kwargs)
    return ri.getMaterial(shelf=shelf, book=book, page=page)


def tabulated_material_from_dep(
    wavelengths_nm: torch.Tensor,
    book: str,
    page=None,
    shelf="main",
    database_path=None,
    dtype=torch.float32,
    device=None,
):
    """
    Build a torch_tmm Material with TabulatedData from dep n,k at the given wavelengths.
    """
    setup_paths()
    mat = material_from_dep(book=book, page=page, shelf=shelf, database_path=database_path)
    nk = load_nk_torch(wavelengths_nm, mat)
    if device is None:
        device = wavelengths_nm.device
    from torch_tmm.dispersion import TabulatedData
    from torch_tmm import Material
    return Material(
        [TabulatedData(wavelengths_nm.to(device), nk.to(device))],
        name=book,
        requires_grad=False,
    )
