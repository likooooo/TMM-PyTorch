"""
测试：仅优化厚度的多种工业场景。
膜系定义在 stack_defs 中，目标函数在 cost_functions 中，优化执行在 run_optimization 中；
本文件只组合“膜系 + 目标 + 配置”并跑多组 case。
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
        stack_single_layer_ar_visible,
        stack_single_layer_ar_center_wavelength,
        stack_two_layer_ar_visible,
        stack_three_layer_broadband_ar,
        stack_single_layer_on_si,
    )
    from .cost_functions import (
        cost_minimize_R_s,
        cost_minimize_R_avg,
        cost_R_at_wavelength,
        cost_maximize_T_s,
        cost_band_R_max,
        cost_band_T_min,
    )
    from .run_optimization import run_optimization
else:
    sys.path.insert(0, _SCRIPT_DIR)
    from stack_defs import (
        stack_single_layer_ar_visible,
        stack_single_layer_ar_center_wavelength,
        stack_two_layer_ar_visible,
        stack_three_layer_broadband_ar,
        stack_single_layer_on_si,
    )
    from cost_functions import (
        cost_minimize_R_s,
        cost_minimize_R_avg,
        cost_R_at_wavelength,
        cost_maximize_T_s,
        cost_band_R_max,
        cost_band_T_min,
    )
    from run_optimization import run_optimization


def case_1_single_layer_ar_minimize_R():
    """Case 1：单层增透（可见光），最小化 s 偏振平均反射率。"""
    stack_def = stack_single_layer_ar_visible()
    result = run_optimization(
        stack_def,
        cost_minimize_R_s,
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 1] 单层增透-最小R  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def case_2_single_layer_ar_center_wavelength():
    """Case 2：单层增透，针对中心波长 550 nm 最小化该波长处 R。"""
    stack_def = stack_single_layer_ar_center_wavelength(center_nm=550.0)
    result = run_optimization(
        stack_def,
        lambda calc, wl, ang: cost_R_at_wavelength(calc, wl, ang, target_wl_nm=550.0),
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 2] 单层中心波长550nm  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def case_3_two_layer_ar_minimize_R_avg():
    """Case 3：双层增透（可见），最小化 s+p 平均反射率。"""
    stack_def = stack_two_layer_ar_visible()
    result = run_optimization(
        stack_def,
        cost_minimize_R_avg,
        opt_config={"max_steps": 1000, "lr": 8e-3, "verbose": True},
    )
    print(f"  [Case 3] 双层增透-最小R_avg  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def case_4_three_layer_broadband_ar():
    """Case 4：三层宽带增透，最小化波段内最大反射率（约束波段内 R 不超标）。"""
    stack_def = stack_three_layer_broadband_ar()
    result = run_optimization(
        stack_def,
        lambda calc, wl, ang: cost_band_R_max(calc, wl, ang, wl_min=450.0, wl_max=650.0),
        opt_config={"max_steps": 1000, "lr": 5e-3, "verbose": True},
    )
    print(f"  [Case 4] 三层宽带-波段内R_max  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def case_5_single_layer_on_si_maximize_T():
    """Case 5：硅基底单层，最大化透射率（等效最小化 1-T）。"""
    stack_def = stack_single_layer_on_si()
    result = run_optimization(
        stack_def,
        cost_maximize_T_s,
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 5] 硅基底单层-最大T  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def case_6_single_layer_ar_band_T_min():
    """Case 6：单层增透，最大化指定波段内最小透射率（保证通带最差点尽量高）。"""
    stack_def = stack_single_layer_ar_visible()
    result = run_optimization(
        stack_def,
        lambda calc, wl, ang: cost_band_T_min(calc, wl, ang, wl_min=500.0, wl_max=600.0),
        opt_config={"max_steps": 1000, "lr": 1e-2, "verbose": True},
    )
    print(f"  [Case 6] 单层-波段内T_min  final loss={result['final_loss']:.6f}  thicknesses={result['thicknesses']}")
    return result


def main():
    print("===== test_optimize_thickness：多 case 仅优化厚度 =====\n")

    cases = [
        ("Case 1 单层增透-最小R(s)", case_1_single_layer_ar_minimize_R),
        ("Case 2 单层中心波长550nm", case_2_single_layer_ar_center_wavelength),
        ("Case 3 双层增透-最小R_avg", case_3_two_layer_ar_minimize_R_avg),
        ("Case 4 三层宽带-波段R_max", case_4_three_layer_broadband_ar),
        ("Case 5 硅基底单层-最大T", case_5_single_layer_on_si_maximize_T),
        ("Case 6 单层-波段内T_min", case_6_single_layer_ar_band_T_min),
    ]

    for name, run_case in cases:
        print(f"\n--- {name} ---")
        try:
            run_case()
        except Exception as e:
            print(f"  FAIL: {e}")
            raise

    print("\n===== test_optimize_thickness 全部通过 =====\n")


if __name__ == "__main__":
    main()
