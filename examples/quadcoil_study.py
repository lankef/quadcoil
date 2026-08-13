"""
quadcoil_study.py
-----------------
Constellaration study script implementing problems A-K.

Usage
-----
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3 --plasma-config-id <id>
    python quadcoil_study.py --input-dir ../output_constellaration_nfp=3 --plasma-coil-distance 1.4
"""

import argparse
import os
import time
from datetime import datetime, timezone
from os.path import join

import numpy as np
import jax
import jax.numpy as jnp

import quadcoil
from quadcoil.quantity import K_theta

jax.config.update("jax_compilation_cache_dir", "../jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def now():
    return datetime.now(timezone.utc).isoformat()


def to_float(x):
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("value", "val", "objective", "metric", "f", "data"):
            if k in x:
                return to_float(x[k])
        return None
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return None


def save_nescoil(path, out_dict, dofs, elapsed, K_theta_avg, K_theta_cons):
    """Save nescoil result. qp is intentionally excluded as it is not serialisable."""
    jnp.save(
        path,
        {
            "out_dict":      out_dict,
            "dofs":          np.array(dofs),
            "time":          elapsed,
            "timestamp_utc": now(),
            "K_theta_avg":   float(K_theta_avg),
            "K_theta_cons":  str(K_theta_cons),
        },
        allow_pickle=True,
    )
    print(f"    -> saved {path}")


def save_result(path, out_dict, dofs, elapsed):
    jnp.save(
        path,
        {
            "out_dict":      out_dict,
            "dofs":          np.array(dofs),
            "time":          elapsed,
            "timestamp_utc": now(),
        },
        allow_pickle=True,
    )
    print(f"    -> saved {path}")


def load_result(path):
    return jnp.load(path, allow_pickle=True).item()


METRICS = (
    "f_B",
    "f_K",
    "f_max_Phi",
    "f_l1_Phi",
    "f_max_K_dot_grad_K_cyl",
    "f_max_K2",
)


def load_quadcoil_inputs(npy_path, mpol, ntor, plasma_coil_distance):
    data = jnp.load(npy_path, allow_pickle=True).item()

    kwargs_base = dict(
        nfp=int(data["nfp"]),
        stellsym=bool(data["stellsym"]),
        mpol=mpol,
        ntor=ntor,
        plasma_dofs=np.array(data["plasma_dofs"]),
        plasma_mpol=4,
        plasma_ntor=4,
        net_poloidal_current_amperes=float(data["net_poloidal_current_amperes"]),
        net_toroidal_current_amperes=0.0,
        plasma_coil_distance=plasma_coil_distance,
    )

    if data.get("Bnormal_plasma") is not None:
        kwargs_base["Bnormal_plasma"] = np.array(data["Bnormal_plasma"])

    return kwargs_base


def run_nescoil(config_dir, quadcoil_kwargs_base):
    """
    Problem A: pure Nescoil (minimise f_B).
    Problem B: Nescoil with K_theta sign constraint.

    qp objects are never saved to disk as they are not serialisable.
    K_theta_avg and K_theta_cons are computed from the live qp after
    problem A and stored in nescoil_A.npy for use on cached runs.
    """
    path_A = join(config_dir, "nescoil_A.npy")
    path_B = join(config_dir, "nescoil_B.npy")

    if not os.path.exists(path_A):
        print("  [A] Nescoil (minimise f_B) ...", flush=True)
        t0 = time.perf_counter()
        out_A, qp_A, dofs_A, _ = quadcoil.quadcoil(
            objective_name="f_B",
            objective_unit=1.0,
            metric_name=METRICS,
            value_only=True,
            **quadcoil_kwargs_base,
        )
        elapsed_A = time.perf_counter() - t0
        print(f"      f_B = {to_float(out_A.get('f_B')):.6g}   ({elapsed_A:.1f} s)")

        K_theta_avg = float(np.average(K_theta(qp_A, dofs_A)))
        K_theta_cons = ">=" if K_theta_avg >= 0 else "<="
        print(f"      K_theta_avg = {K_theta_avg:.4g}  -> K_theta {K_theta_cons} 0")

        save_nescoil(path_A, out_A, dofs_A, elapsed_A, K_theta_avg, K_theta_cons)
    else:
        print("  [A] cached", flush=True)
        d = load_result(path_A)
        out_A = d["out_dict"]
        K_theta_avg = float(d["K_theta_avg"])
        K_theta_cons = str(d["K_theta_cons"])
        print(f"      f_B = {to_float(out_A.get('f_B')):.6g}")
        print(f"      K_theta_avg = {K_theta_avg:.4g}  -> K_theta {K_theta_cons} 0")

    fB_A = to_float(out_A.get("f_B"))
    K_theta_unit = max(abs(K_theta_avg), 1.0)

    if not os.path.exists(path_B):
        print("  [B] Nescoil w/ K_theta sign constraint ...", flush=True)
        t0 = time.perf_counter()
        out_B, qp_B, dofs_B, _ = quadcoil.quadcoil(
            objective_name="f_B",
            objective_unit=1.0,
            constraint_name=("K_theta",),
            constraint_type=(K_theta_cons,),
            constraint_value=np.array([0.0]),
            constraint_unit=(K_theta_unit,),
            metric_name=METRICS,
            value_only=True,
            **quadcoil_kwargs_base,
        )
        elapsed_B = time.perf_counter() - t0
        print(f"      f_B = {to_float(out_B.get('f_B')):.6g}   ({elapsed_B:.1f} s)")
        save_nescoil(path_B, out_B, dofs_B, elapsed_B, K_theta_avg, K_theta_cons)
    else:
        print("  [B] cached", flush=True)
        d = load_result(path_B)
        out_B = d["out_dict"]
        print(f"      f_B = {to_float(out_B.get('f_B')):.6g}")

    fB_B = to_float(out_B.get("f_B"))
    return fB_A, fB_B, K_theta_cons, K_theta_avg


def build_fB_targets(fB_A, fB_B, n=5):
    scale = 10 ** np.linspace(0, 1, n)
    return scale * fB_A, scale * fB_B


def make_kwargs_list(K_theta_cons, K_theta_avg):
    K_theta_unit = max(abs(K_theta_avg), 1.0)

    filament_extra = (
        ("K_theta",),
        (K_theta_cons,),
        np.array([0.0]),
        (K_theta_unit,),
    )

    def dipole(label, obj):
        return dict(label=label, target_key="dipole",
                    objective_name=obj, extra_constraints=None)

    def filament(label, obj):
        return dict(label=label, target_key="filament",
                    objective_name=obj, extra_constraints=filament_extra)

    return [
        dipole("C_sparse_dipole",        "f_l1_Phi"),
        dipole("D_thin_dipole",          "f_max_Phi"),
        dipole("F_low_max_force_dipole", "f_max_force_cyl"),
        filament("I_low_curvature_filament", "f_max_K_dot_grad_K_cyl"),
        filament("K_low_max_force_filament", "f_max_force_cyl"),
    ]


def run_one_config(config_dir, quadcoil_kwargs_base, n_targets=5):
    fB_A, fB_B, K_theta_cons, K_theta_avg = run_nescoil(config_dir, quadcoil_kwargs_base)

    print(f"  fB_A={fB_A:.4g}  fB_B={fB_B:.4g}")

    fB_target_dipole, fB_target_filament = build_fB_targets(fB_A, fB_B, n=n_targets)
    target_arrays = {"dipole": fB_target_dipole, "filament": fB_target_filament}
    obj_units     = {"dipole": fB_A,             "filament": fB_B}

    kwargs_list = make_kwargs_list(K_theta_cons, K_theta_avg)

    for kwarg_i in kwargs_list:
        label      = kwarg_i["label"]
        target_key = kwarg_i["target_key"]
        obj_name   = kwarg_i["objective_name"]
        obj_unit   = obj_units[target_key]
        fB_targets = target_arrays[target_key]
        extra      = kwarg_i["extra_constraints"]

        print(f"\n  [{label}]  objective={obj_name}", flush=True)

        for i in range(n_targets):
            fB_target_i = float(fB_targets[i])
            save_path   = join(config_dir, f"{label}_i{i}.npy")

            if os.path.exists(save_path):
                print(f"    i={i}: cached, skipping.")
                continue

            print(f"    i={i}  fB_target={fB_target_i:.4g}", end="  ", flush=True)

            if extra is None:
                c_names  = ("f_B",)
                c_types  = ("<=",)
                c_values = np.array([fB_target_i])
                c_units  = (fB_target_i,)
            else:
                ex_names, ex_types, ex_values, ex_units = extra
                c_names  = ("f_B",)  + ex_names
                c_types  = ("<=",)   + ex_types
                c_values = np.concatenate([[fB_target_i], ex_values])
                c_units  = (fB_target_i,) + ex_units

            try:
                t0 = time.perf_counter()
                out_dict, qp, dofs, status = quadcoil.quadcoil(
                    objective_name=obj_name,
                    objective_unit=obj_unit,
                    constraint_name=c_names,
                    constraint_type=c_types,
                    constraint_value=c_values,
                    constraint_unit=c_units,
                    metric_name=METRICS,
                    value_only=True,
                    **quadcoil_kwargs_base,
                )
                elapsed = time.perf_counter() - t0

                note = " (JIT compile + run)" if i == 0 else ""
                print(f"({elapsed:.1f} s{note})")
                print(f"      f_B={to_float(out_dict.get('f_B')):.4g}  "
                      f"fin_f={status.get('inner_fin_f', '?'):.4f}  "
                      f"niter={status.get('inner_fin_niter', '?')}")

                save_result(save_path, out_dict, dofs, elapsed)

            except Exception as e:
                print(f"\n      FAILED: {e}")


def find_config_dirs(input_dir):
    configs = []
    for name in sorted(os.listdir(input_dir)):
        subdir = join(input_dir, name)
        if os.path.isdir(subdir) and os.path.exists(join(subdir, "quadcoil_inputs.npy")):
            configs.append((name, subdir))
    return configs


def run_study(input_dir, mpol, ntor, n_targets, plasma_coil_distance, plasma_config_id=None, task_id=0, num_tasks=1):
    configs = find_config_dirs(input_dir)

    if not configs:
        raise RuntimeError(f"No quadcoil_inputs.npy files found under {input_dir}")

    if plasma_config_id is not None:
        configs = [(n, d) for n, d in configs if n == plasma_config_id]
        if not configs:
            raise RuntimeError(f"plasma_config_id '{plasma_config_id}' not found in {input_dir}")
    else:
        configs = [c for i, c in enumerate(configs) if i % num_tasks == task_id]

    print(f"Found {len(configs)} config(s) to process.")

    for idx, (config_id, config_dir) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Config {idx+1}/{len(configs)}: {config_id}")
        print(f"{'='*60}")

        npy_path = join(config_dir, "quadcoil_inputs.npy")
        try:
            quadcoil_kwargs_base = load_quadcoil_inputs(
                npy_path, mpol, ntor, plasma_coil_distance
            )
        except Exception as e:
            print(f"  Failed to load inputs: {e} -- skipping.")
            continue

        try:
            run_one_config(config_dir, quadcoil_kwargs_base, n_targets=n_targets)
        except Exception as e:
            print(f"  Config failed: {e} -- continuing to next.")

    print("\n\nAll configs complete.")


def main():
    p = argparse.ArgumentParser(
        description="quadcoil constellaration study - problems A through K"
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--plasma-config-id", default=None)
    p.add_argument("--mpol",      type=int,   default=8)
    p.add_argument("--ntor",      type=int,   default=8)
    p.add_argument("--n-targets", type=int,   default=5)
    p.add_argument("--plasma-coil-distance", type=float, default=1.4)
    p.add_argument("--task-id",   type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    args = p.parse_args()

    run_study(
        input_dir=args.input_dir,
        mpol=args.mpol,
        ntor=args.ntor,
        n_targets=args.n_targets,
        plasma_coil_distance=args.plasma_coil_distance,
        plasma_config_id=args.plasma_config_id,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()