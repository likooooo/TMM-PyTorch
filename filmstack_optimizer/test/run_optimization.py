"""
优化执行模块：将膜系定义与优化配置解耦，根据 stack_def 构建优化器并运行。
不包含具体膜系或目标函数定义，仅负责“给定膜系 + 目标函数 + 优化配置 → 运行并返回结果”。
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

from filmstack_optimizer.core import setup_paths, FilmStackOptimizer
from filmstack_optimizer.core.constraints import (
    thickness_from_raw,
    nk_from_raw,
    THICKNESS_MIN_NM,
    THICKNESS_MAX_NM,
)

setup_paths()


def run_optimization(
    stack_def: dict,
    cost_fn,
    opt_config: dict = None,
):
    """
    根据膜系定义与目标函数执行优化，解耦“建堆栈”与“做优化”。

    参数
    ----------
    stack_def : dict
        来自 stack_defs 的字典，需包含：
        wavelengths, angles, env_layer, subs_layer, structure_spec,
        dtype, device。可选 thickness_min, thickness_max 作全局厚度默认。
        structure_spec 每层可含独立范围：thickness_min, thickness_max；
        可学习 n,k 层可含 n_min, n_max, k_min, k_max（四者独立）。
    cost_fn : callable
        cost_fn(calc, wavelengths, angles) -> 标量 Tensor。
        计算时会用 stack_def 中的 wavelengths/angles 传入。
    opt_config : dict, optional
        steps / max_steps, min_steps, lr, grad_clip, verbose,
        early_stop_patience, early_stop_tol,
        thickness_min, thickness_max（全局厚度范围，默认 0 / 5000 nm），
        plot_convergence（默认 True）, plot_save_path（可选）。
        默认：min_steps=100, max_steps=1000（若传 steps 则 max_steps 取 steps，否则 1000）,
        early_stop_patience=25, early_stop_tol=0（默认关闭提前退出；>0 时启用）。
    stack_def 中也可含 thickness_min, thickness_max 作为全局默认；每层 structure_spec 项可含 thickness_min, thickness_max 覆盖该层。

    返回
    ----------
    dict
        history: 每步 loss 列表；
        final_loss: 最后一步 loss；
        thicknesses: 各层最终厚度（list of float，nm）；
        nk: 各层 (n,k) 或 None（list）；
        filmstack_instruction: 多层膜构建指令字符串（厚度 μm，可直接用于 Fresnel 计算器）；
        performance: 最终膜系性能 dict（R_s_mean, R_p_mean, T_s_mean, T_p_mean, R_avg_mean, T_avg_mean, R_s_max, R_p_max, T_s_max, T_p_max）；
        optimizer: FilmStackOptimizer 实例（便于进一步查询）。
    """
    opt_config = opt_config or {}
    # 最大迭代：优先 max_steps，否则用 steps（兼容旧用法），默认 1000
    max_steps = opt_config.get("max_steps", opt_config.get("steps", 1000))
    min_steps = opt_config.get("min_steps", 100)
    early_stop_patience = opt_config.get("early_stop_patience", 25)
    early_stop_tol = opt_config.get("early_stop_tol", 0)
    lr = opt_config.get("lr", 1e-2)
    grad_clip = opt_config.get("grad_clip", 1.0)
    verbose = opt_config.get("verbose", True)
    plot_convergence = opt_config.get("plot_convergence", True)
    plot_save_path = opt_config.get("plot_save_path")

    wl = stack_def["wavelengths"]
    angles = stack_def["angles"]
    env = stack_def["env_layer"]
    subs = stack_def["subs_layer"]
    structure_spec = stack_def["structure_spec"]
    dtype = stack_def["dtype"]
    device = stack_def["device"]
    thickness_min = stack_def.get("thickness_min", opt_config.get("thickness_min", THICKNESS_MIN_NM))
    thickness_max = stack_def.get("thickness_max", opt_config.get("thickness_max", THICKNESS_MAX_NM))

    def wrapped_loss(calc):
        return cost_fn(calc, wl, angles)

    opt = FilmStackOptimizer(
        wavelengths=wl,
        angles=angles,
        env_layer=env,
        subs_layer=subs,
        structure_spec=structure_spec,
        thickness_min=thickness_min,
        thickness_max=thickness_max,
        dtype=dtype,
        device=device,
    )

    history = opt.run(
        steps=max_steps,
        loss_fn=wrapped_loss,
        lr=lr,
        grad_clip=grad_clip,
        verbose=verbose,
        min_steps=min_steps,
        max_steps=max_steps,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
    )

    thicknesses = [
        thickness_from_raw(
            opt._thickness_raws[i],
            thickness_min=opt._thickness_mins[i],
            thickness_max=opt._thickness_maxs[i],
        ).item()
        for i in range(len(opt._thickness_raws))
    ]
    nk_list = []
    for i in range(len(structure_spec)):
        if opt._fixed_materials[i] is not None:
            nk_list.append(None)
        else:
            n_val = nk_from_raw(
                opt._n_raws[i],
                nk_min=opt._n_mins[i],
                nk_max=opt._n_maxs[i],
            ).item()
            k_val = nk_from_raw(
                opt._k_raws[i],
                nk_min=opt._k_mins[i],
                nk_max=opt._k_maxs[i],
            ).item()
            nk_list.append((n_val, k_val))

    # 构建指令用 n/k：固定材料在参考波长处取 n、k，可学习层用优化结果
    wl_ref = wl[len(wl) // 2 : len(wl) // 2 + 1] if len(wl) > 0 else None  # (1,) 保证 refractive_index 返回 1D
    nk_for_instruction = []
    for i in range(len(structure_spec)):
        if opt._fixed_materials[i] is not None:
            if wl_ref is not None:
                n_complex = opt._fixed_materials[i].refractive_index(wl_ref)
                nr = n_complex.real.squeeze().item()
                ni = n_complex.imag.squeeze().item()
                nk_for_instruction.append((nr, ni))
            else:
                nk_for_instruction.append((0.0, 0.0))
        else:
            nk_for_instruction.append(nk_list[i])

    final_loss = history[-1] if history else float("nan")

    # 最终膜系性能：用当前参数做一次前向得到 R、T
    calc_final = opt.forward()
    performance = _compute_performance(calc_final)

    filmstack_instruction = format_filmstack_instruction(structure_spec, thicknesses, nk_for_instruction)
    if opt_config.get("verbose", True):
        print("\n  [多层膜构建指令] (厚度 μm，n k)")
        print(f"  {filmstack_instruction}")
        _print_performance(performance)

    if plot_convergence and history:
        _plot_convergence(history, stack_def.get("name", "optimization"), plot_save_path)

    return {
        "history": history,
        "final_loss": final_loss,
        "thicknesses": thicknesses,
        "nk": nk_list,
        "filmstack_instruction": filmstack_instruction,
        "performance": performance,
        "optimizer": opt,
    }


def _compute_performance(calc):
    """从 OpticalCalculator 计算膜系性能：R_s, R_p, T_s, T_p 及平均（在波长×角度上取均值）。"""
    R_s = calc.reflection("s").detach().cpu()
    R_p = calc.reflection("p").detach().cpu()
    T_s = calc.transmission("s").detach().cpu()
    T_p = calc.transmission("p").detach().cpu()
    R_s = R_s.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    R_p = R_p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    T_s = T_s.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    T_p = T_p.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return {
        "R_s_mean": float(R_s.mean()),
        "R_p_mean": float(R_p.mean()),
        "T_s_mean": float(T_s.mean()),
        "T_p_mean": float(T_p.mean()),
        "R_avg_mean": float((R_s + R_p).mean() / 2),
        "T_avg_mean": float((T_s + T_p).mean() / 2),
        "R_s_max": float(R_s.max()),
        "R_p_max": float(R_p.max()),
        "T_s_max": float(T_s.max()),
        "T_p_max": float(T_p.max()),
    }


def _print_performance(performance):
    """打印性能参数字段。"""
    print("  [膜系性能] (波长×角度平均)")
    print(f"    R_s = {performance['R_s_mean']:.6f}   R_p = {performance['R_p_mean']:.6f}   R_avg = {performance['R_avg_mean']:.6f}")
    print(f"    T_s = {performance['T_s_mean']:.6f}   T_p = {performance['T_p_mean']:.6f}   T_avg = {performance['T_avg_mean']:.6f}")
    print(f"    R_s_max = {performance['R_s_max']:.6f}   R_p_max = {performance['R_p_max']:.6f}")
    print(f"    T_s_max = {performance['T_s_max']:.6f}   T_p_max = {performance['T_p_max']:.6f}")


def format_filmstack_instruction(structure_spec, thicknesses, nk_list):
    """
    将优化结果格式化为 Fresnel 多层膜构建指令字符串（参考 simulation_toykits 文档）。
    格式：Vacuum 0 <层1 厚度_μm n k> <层2 ...> ... Vacuum 0
    厚度单位 μm，每层都输出 n、k（固定材料为参考波长处的值，可学习层为优化结果）。
    nk_list 应为与 structure_spec 等长、每项为 (n, k) 的列表。
    """
    parts = ["Vacuum", "0"]
    for i, spec in enumerate(structure_spec):
        name = None
        if "material" in spec and hasattr(spec["material"], "name"):
            name = spec["material"].name
        if not name:
            name = spec.get("layer_name", f"Layer{i+1}")
        d_nm = thicknesses[i] if i < len(thicknesses) else 0.0
        d_um = d_nm / 1000.0
        nk = nk_list[i] if i < len(nk_list) else (0.0, 0.0)
        n_val, k_val = nk
        parts.append(f"{name} {d_um:.6g} {n_val:.6g} {k_val:.6g}")
    parts.extend(["Vacuum", "0"])
    return " ".join(parts)


def _plot_convergence(history, name="optimization", save_path=None):
    """绘制收敛曲线：loss vs step。若 save_path 为 None 则自动生成文件名并保存到当前目录或临时路径。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # 过滤 nan 以便绘图（保留索引对应 step）
    steps = list(range(1, len(history) + 1))
    values = [h if (h == h and abs(h) != float("inf")) else None for h in history]
    valid = [(s, v) for s, v in zip(steps, values) if v is not None]
    if not valid:
        return
    xs, ys = zip(*valid)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(xs, ys, "b-", linewidth=1.5, label="loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence: {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is None:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))
        save_path = f"convergence_{safe_name}.png"
    try:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close(fig)
