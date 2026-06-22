"""
Performance benchmark comparing legacy vs autodiff unitnormaldash implementations.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
from jax import block_until_ready
import time
from load_test_data import load_data


def benchmark_unitnormaldash():
    """Compare performance of legacy vs autodiff unitnormaldash implementations."""
    print("=" * 70)
    print("Performance Benchmark: unitnormaldash_legacy vs unitnormaldash")
    print("=" * 70)
    
    # Load test surface
    print("\n1. Loading test surface...")
    _, _, _, _, qp = load_data()
    base_surface = qp.plasma_surface
    nphi = len(base_surface.quadpoints_phi)
    ntheta = len(base_surface.quadpoints_theta)
    print(f"   Surface shape: {nphi} x {ntheta}")
    
    # Warm up JIT compilation
    print("\n2. Warming up JIT compilation...")
    print("   Compiling legacy implementation...")
    _ = base_surface.unitnormaldash_legacy()
    block_until_ready(_)
    print("   ✓ Legacy compiled")
    
    print("   Compiling autodiff implementation...")
    _ = base_surface.unitnormaldash(1, 0)
    block_until_ready(_)
    _ = base_surface.unitnormaldash(0, 1)
    block_until_ready(_)
    print("   ✓ Autodiff compiled")
    
    # Benchmark legacy implementation
    print("\n3. Benchmarking legacy implementation...")
    n_runs = 100
    times_legacy = []
    
    for i in range(n_runs):
        start = time.perf_counter()
        result = base_surface.unitnormaldash_legacy()
        block_until_ready(result)
        end = time.perf_counter()
        times_legacy.append((end - start) * 1000)  # Convert to ms
    
    mean_legacy = jnp.mean(jnp.array(times_legacy))
    std_legacy = jnp.std(jnp.array(times_legacy))
    print(f"   Mean time: {mean_legacy:.4f} ± {std_legacy:.4f} ms")
    
    # Benchmark autodiff implementation
    print("\n4. Benchmarking autodiff implementation...")
    times_autodiff = []
    
    for i in range(n_runs):
        start = time.perf_counter()
        result1 = base_surface.unitnormaldash(1, 0)
        result2 = base_surface.unitnormaldash(0, 1)
        block_until_ready(result1)
        block_until_ready(result2)
        end = time.perf_counter()
        times_autodiff.append((end - start) * 1000)  # Convert to ms
    
    mean_autodiff = jnp.mean(jnp.array(times_autodiff))
    std_autodiff = jnp.std(jnp.array(times_autodiff))
    print(f"   Mean time: {mean_autodiff:.4f} ± {std_autodiff:.4f} ms")
    
    # Compare results
    print("\n5. Comparing results...")
    unitnormaldash1_legacy, unitnormaldash2_legacy = base_surface.unitnormaldash_legacy()
    unitnormaldash1_autodiff = base_surface.unitnormaldash(1, 0)
    unitnormaldash2_autodiff = base_surface.unitnormaldash(0, 1)
    
    max_diff_phi = jnp.max(jnp.abs(unitnormaldash1_legacy - unitnormaldash1_autodiff))
    max_diff_theta = jnp.max(jnp.abs(unitnormaldash2_legacy - unitnormaldash2_autodiff))
    
    print(f"   Max difference (phi):   {max_diff_phi:.2e}")
    print(f"   Max difference (theta): {max_diff_theta:.2e}")
    
    if max_diff_phi < 1e-10 and max_diff_theta < 1e-10:
        print("   ✓ Results are numerically equivalent")
    else:
        print("   ⚠ Results differ!")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Legacy implementation:   {mean_legacy:.4f} ± {std_legacy:.4f} ms")
    print(f"Autodiff implementation: {mean_autodiff:.4f} ± {std_autodiff:.4f} ms")
    print(f"Slowdown factor:         {mean_autodiff / mean_legacy:.2f}x")
    print("=" * 70)
    
    # Test second derivatives (only available with autodiff)
    print("\n6. Testing second derivatives (autodiff only)...")
    
    start = time.perf_counter()
    d2_phi2 = base_surface.unitnormaldash(2, 0)
    block_until_ready(d2_phi2)
    end = time.perf_counter()
    time_d2_phi2 = (end - start) * 1000
    print(f"   d²(unitnormal)/dphi²:       {time_d2_phi2:.4f} ms")
    
    start = time.perf_counter()
    d2_theta2 = base_surface.unitnormaldash(0, 2)
    block_until_ready(d2_theta2)
    end = time.perf_counter()
    time_d2_theta2 = (end - start) * 1000
    print(f"   d²(unitnormal)/dtheta²:     {time_d2_theta2:.4f} ms")
    
    start = time.perf_counter()
    d2_mixed = base_surface.unitnormaldash(1, 1)
    block_until_ready(d2_mixed)
    end = time.perf_counter()
    time_d2_mixed = (end - start) * 1000
    print(f"   d²(unitnormal)/dphi dtheta: {time_d2_mixed:.4f} ms")
    
    print("\n✅ Benchmark complete!")
    
    return {
        'mean_legacy': float(mean_legacy),
        'std_legacy': float(std_legacy),
        'mean_autodiff': float(mean_autodiff),
        'std_autodiff': float(std_autodiff),
        'slowdown_factor': float(mean_autodiff / mean_legacy),
        'max_diff_phi': float(max_diff_phi),
        'max_diff_theta': float(max_diff_theta),
        'time_d2_phi2': float(time_d2_phi2),
        'time_d2_theta2': float(time_d2_theta2),
        'time_d2_mixed': float(time_d2_mixed),
    }


if __name__ == "__main__":
    results = benchmark_unitnormaldash()
