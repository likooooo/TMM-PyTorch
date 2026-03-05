# Film Stack Optimizer

Multilayer film optimizer based on the **Transfer Matrix Method (TMM)** and **PyTorch** autograd. It treats transmittance/reflectance as continuous functions of layer thickness and refractive index (n, k), and supports **inequality constraints** via reparametrization.

## Features

- **Optimization variables**: layer thickness and/or n, k per layer.
- **Inequality constraints** (enforced by reparametrization):
  - Thickness ≥ 0
  - 0 ≤ n ≤ 10, 0 ≤ k ≤ 10
- **Compatibility**: uses `torch_tmm` (TMM-PyTorch) for the forward pass and can use `dep/refractiveindex.py` for n,k data.

## Layout

- **`core/`** – optimizer implementation:
  - `path_setup.py`: adds repo root, `artifacts`, and `libs` to `sys.path` so `torch_tmm` and `dep` are importable.
  - `constraints.py`: reparametrization for bounds (softplus for thickness, sigmoid for n/k).
  - `learnable_nk.py`: learnable n,k dispersion compatible with `torch_tmm.Material`.
  - `optimizer.py`: `FilmStackOptimizer` – builds TMM stack from raw parameters and runs optimization.
  - `nk_from_dep.py`: load n,k from `dep/refractiveindex.py` and convert to torch (for use with tabulated materials).
- **`test/`** – 测试（膜系与优化解耦）:
  - **`stack_defs.py`**：膜系定义（仅构建堆栈），如单层/双层/三层增透、硅基底、可优化 n,k 等工业常用结构。
  - **`cost_functions.py`**：目标函数（损失），如最小化/最大化 R 或 T、单波长 R、波段内 R_max/T_min、匹配目标曲线等。
  - **`run_optimization.py`**：根据 `stack_def` + `cost_fn` + `opt_config` 执行优化，返回 history、厚度、nk 等。
  - **`test_optimize_thickness.py`**：多组仅优化厚度的 case（单层/双层/三层增透、中心波长、硅基底、波段约束等）。
  - **`test_optimize_nk.py`**：多组优化 n、k（及厚度）的 case（最小 R、最大 R、单波长、匹配目标曲线等）。
- **`dep/`** – optional dependency; `refractiveindex.py` for n,k lookup (compatible data structures).

## Setup

1. **Install dependencies** (from repo root or from `filmstack_optimizer/`):

   ```bash
   pip install -r filmstack_optimizer/requirements.txt
   ```

2. **Activate your environment** (if you use conda/venv):

   ```bash
   source activate <your_env>
   ```

3. **Path dependency**: `core` expects the **TMM-PyTorch repo root** and, if present, **`artifacts`** and **`libs`** directories to be on `sys.path` so that `torch_tmm` can be imported. `path_setup.setup_paths()` is called by the optimizer and adds the repo root (and `artifacts`, `libs` when they exist). Run tests from the **repo root** so that `filmstack_optimizer` and the repo root are on the path.

## Usage

### Optimize thickness only

```python
from filmstack_optimizer.core import setup_paths, FilmStackOptimizer
setup_paths()

import torch
from torch_tmm import Material, Layer, Dispersion

wavelengths = torch.linspace(450, 650, 41)
angles = torch.tensor([0.0])
env = Layer(Material([Dispersion.Constant_epsilon(1.0)], name="Air"), layer_type="semi-inf")
subs = Layer(Material([Dispersion.Constant_epsilon(1.5**2)], name="Sub"), layer_type="semi-inf")
structure_spec = [{"thickness": 100.0, "material": Material([Dispersion.Constant_epsilon(1.38**2)], name="L")}]

opt = FilmStackOptimizer(wavelengths, angles, env, subs, structure_spec)
def loss_fn(calc):
    return calc.reflection("s").mean()
opt.run(steps=150, loss_fn=loss_fn)
# opt.get_thicknesses() gives current thicknesses
```

### Optimize thickness and n, k

```python
structure_spec = [{"thickness": 80.0, "n": 1.5, "k": 0.01}]  # no "material" -> learnable n,k；可加 thickness_min/max, n_min/n_max, k_min/k_max 每层独立
opt = FilmStackOptimizer(wavelengths, angles, env, subs, structure_spec)
opt.run(steps=120, loss_fn=loss_fn)
# opt.get_thicknesses(), opt.get_nk() for current values
```

### Using dep for n,k

```python
from filmstack_optimizer.core.nk_from_dep import tabulated_material_from_dep

# Build a torch_tmm Material from refractiveindex.info (book/page/shelf)
wl = torch.linspace(400, 800, 81)
mat = tabulated_material_from_dep(wl, book="SiO2", shelf="main")  # optional: page=..., database_path=...
# Use mat in structure_spec: {"thickness": 50.0, "material": mat}
```

## Running tests

测试将**膜系定义**与**优化执行**解耦：膜系在 `stack_defs.py`，目标在 `cost_functions.py`，运行逻辑在 `run_optimization.py`；两个测试文件只组合多组 (膜系, 目标, 配置) case。

从 **TMM-PyTorch 仓库根目录**执行（已安装 `filmstack_optimizer/requirements.txt` 并激活环境）：

```bash
python -m filmstack_optimizer.test.test_optimize_thickness
python -m filmstack_optimizer.test.test_optimize_nk
```

或直接运行脚本（需保证仓库根在 `PYTHONPATH` 或当前目录为仓库根）：

```bash
python filmstack_optimizer/test/test_optimize_thickness.py
python filmstack_optimizer/test/test_optimize_nk.py
```

**厚度测试**包含多组工业常见 case：单层增透（最小 R）、单层中心波长 550 nm、双层/三层增透、硅基底单层最大 T、波段内 R_max/T_min 等。**n/k 测试**包含：可学习 nk 最小 R、最大 R、550 nm 处 R、匹配目标反射率曲线、最小 R_avg 等。

## Constraints

- **厚度**：每层独立范围 `[thickness_min, thickness_max]`（默认 0 / 5000 nm）。`thickness = thickness_min + (thickness_max - thickness_min) * sigmoid(raw)`。在 `structure_spec` 每层可设 `thickness_min`, `thickness_max`，未设则用全局默认。
- **n, k**：可学习 n、k 的层中，n 与 k **各自独立**范围 `[n_min, n_max]`、`[k_min, k_max]`（默认 0 / 10）。在 `structure_spec` 该层可设 `n_min`, `n_max`, `k_min`, `k_max` 四个参数，未设则用默认。

## 关于优化出现 NaN 的说明与修复

**原因概览：**

1. **厚度参数化溢出**：原先用 `raw = log(expm1(thickness))` 初始化。在 float32 下 `exp(100)` 已超出表示范围，`expm1(100)` 为 inf，`log(inf)` 为 inf，导致厚度与 TMM 内部出现 inf/NaN。
2. **反射率除零/奇异**：反射系数 `r = T[1,0]/T[0,0]`。当某些波长/厚度下 `T[0,0]` 接近 0 时，`r` 和梯度会极大，引发梯度爆炸和 NaN。
3. **厚度过大**：厚度过大时 TMM 传播矩阵中 `exp(delta)` 可能溢出。

**已做修改：**

- **constraints.py**：厚度改为 `thickness = exp(raw)` 并夹紧到最大 5000 nm；初始化用 `raw = log(thickness)`，不再使用会溢出的 `log(expm1(t))`。
- **optimizer.py**：默认损失中对反射率做 `clamp(0, 2)` 和 `nan_to_num`；`run()` 中增加梯度裁剪（`clip_grad_norm_`，默认 max_norm=1）；若某步 loss 为 nan/inf 则跳过该步不更新参数。
- 测试中改为使用优化器默认损失（带夹紧与梯度裁剪），避免自定义损失未做稳定性处理。

自定义 `loss_fn` 时建议对反射率/透射率做夹紧并对 loss 做 `nan_to_num`，并设置 `run(..., grad_clip=1.0)`。
