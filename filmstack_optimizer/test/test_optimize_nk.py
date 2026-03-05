"""
测试：优化 n、k（及厚度）的多种工业场景。
膜系定义在 stack_defs，目标函数在 cost_functions，优化在 run_optimization；
本文件只组合不同 case 并运行。
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

if __package__:
    from .stack_defs import (
        stack_learnable_nk_single_ar,
        stack_learnable_nk_single_high_r,
        stack_learnable_nk_single_match_target,
    )
    from .cost_functions import (
        cost_minimize_R_s,
        cost_minimize_R_avg,
        cost_maximize_R_s,
        cost_R_at_wavelength,
        cost_match_target_R,
    )
    from .run_optimization import run_optimization
else:
    sys.path.insert(0, _SCRIPT_DIR)
    from stack_defs import (
        stack_learnable_nk_single_ar,
        stack_learnable_nk_single_high_r,
        stack_learnable_nk_single_match_target,
    )
    from cost_functions import (
        cost_minimize_R_s,
        cost_minimize_R_avg,
        cost_maximize_R_s,
        cost_R_at_wavelength,
        cost_match_target_R,
    )
    from run_optimization import run_optimization


def case_1_learnable_nk_minimize_R():
    """Case 1：单层可优化 n、k，最小化平均反射率（增透型初值）。"""
    stack_def = stack_learnable_nk_single_ar()
    result = run_optimization(
        stack_def,
        cost_minimize_R_s,
        opt_config={"max_steps": 1000, "lr": 2e-2, "verbose": True},
    )
    print(f"  [Case 1] 可学习nk-最小R  loss={result['final_loss']:.6f}  d={result['thicknesses']}  nk={result['nk']}")
    return result


def case_2_learnable_nk_maximize_R():
    """Case 2：单层可优化 n、k，最大化反射率（高反/金属型初值）。"""
    stack_def = stack_learnable_nk_single_high_r()
    result = run_optimization(
        stack_def,
        cost_maximize_R_s,
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 2] 可学习nk-最大R  loss={result['final_loss']:.6f}  d={result['thicknesses']}  nk={result['nk']}")
    return result


def case_3_learnable_nk_R_at_550():
    """Case 3：单层可优化 n、k，最小化 550 nm 处反射率。"""
    stack_def = stack_learnable_nk_single_match_target()
    result = run_optimization(
        stack_def,
        lambda calc, wl, ang: cost_R_at_wavelength(calc, wl, ang, target_wl_nm=550.0),
        opt_config={"max_steps": 1000, "lr": 2e-2, "verbose": True},
    )
    print(f"  [Case 3] 可学习nk-550nm处R  loss={result['final_loss']:.6f}  d={result['thicknesses']}  nk={result['nk']}")
    return result


def case_4_learnable_nk_match_target_curve():
    """Case 4：单层可优化 n、k，匹配目标反射率曲线（例如目标为 5% 平坦）。"""
    import torch
    stack_def = stack_learnable_nk_single_match_target()
    wl = stack_def["wavelengths"]
    # 目标：波长范围内反射率约 0.05
    target_R = torch.full_like(wl, 0.05, device=wl.device, dtype=wl.dtype)
    result = run_optimization(
        stack_def,
        lambda calc, w, a: cost_match_target_R(calc, w, a, target_R=target_R, pol="s"),
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 4] 可学习nk-匹配R=0.05  loss={result['final_loss']:.6f}  d={result['thicknesses']}  nk={result['nk']}")
    return result


def case_5_learnable_nk_minimize_R_avg():
    """Case 5：单层可优化 n、k，最小化 s+p 平均反射率。"""
    stack_def = stack_learnable_nk_single_ar()
    result = run_optimization(
        stack_def,
        cost_minimize_R_avg,
        opt_config={"max_steps": 1000, "lr": 2e-2, "verbose": True},
    )
    print(f"  [Case 5] 可学习nk-最小R_avg  loss={result['final_loss']:.6f}  d={result['thicknesses']}  nk={result['nk']}")
    return result


def main():
    print("===== test_optimize_nk：多 case 优化 n、k（及厚度） =====\n")

    cases = [
        ("Case 1 可学习nk-最小R", case_1_learnable_nk_minimize_R),
        ("Case 2 可学习nk-最大R", case_2_learnable_nk_maximize_R),
        ("Case 3 可学习nk-550nm处R", case_3_learnable_nk_R_at_550),
        ("Case 4 可学习nk-匹配R=0.05", case_4_learnable_nk_match_target_curve),
        ("Case 5 可学习nk-最小R_avg", case_5_learnable_nk_minimize_R_avg),
    ]

    for name, run_case in cases:
        print(f"\n--- {name} ---")
        try:
            run_case()
        except Exception as e:
            print(f"  FAIL: {e}")
            raise

    print("\n===== test_optimize_nk 全部通过 =====\n")


if __name__ == "__main__":
    main()
