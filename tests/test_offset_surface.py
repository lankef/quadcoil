#!/usr/bin/env python3
"""
Test SurfaceOffsetJAX using real test data.
"""

import sys
sys.path.insert(0, '/home/lf2869/Documents/Codes/quadcoil/src')
sys.path.insert(0, '/home/lf2869/Documents/Codes/quadcoil/tests')

import jax.numpy as jnp
from load_test_data import load_data
from quadcoil.surface import SurfaceOffsetJAX

def test_with_real_surface():
    """Test offset surface with real surface from test data."""
    print("Loading test data...")
    _, _, _, _, qp = load_data()
    
    base_surface = qp.plasma_surface
    print(f"✓ Loaded plasma surface: nfp={base_surface.nfp}, mpol={base_surface.mpol}, ntor={base_surface.ntor}")
    print(f"  Grid: {len(base_surface.quadpoints_phi)} × {len(base_surface.quadpoints_theta)}")
    
    # Test that base surface is valid
    gamma_base = base_surface.gamma()
    print(f"  Base gamma shape: {gamma_base.shape}, has NaN: {jnp.any(jnp.isnan(gamma_base))}")
    assert not jnp.any(jnp.isnan(gamma_base)), "Base surface has NaN values!"
    
    # Create offset surface
    offset_distance = 0.1
    offset_surface = SurfaceOffsetJAX(base_surface, offset_distance)
    print(f"✓ Created offset surface with d={offset_distance}")
    
    # Test gamma
    gamma_offset = offset_surface.gamma()
    unitnormal_base = base_surface.unitnormal()
    gamma_expected = gamma_base + offset_distance * unitnormal_base
    
    print(f"  Offset gamma has NaN: {jnp.any(jnp.isnan(gamma_offset))}")
    print(f"  Expected gamma has NaN: {jnp.any(jnp.isnan(gamma_expected))}")
    
    max_diff = jnp.max(jnp.abs(gamma_offset - gamma_expected))
    print(f"  Max difference: {max_diff}")
    assert jnp.allclose(gamma_offset, gamma_expected, atol=1e-10)
    print("✓ gamma() offset correctly")
    
    # Test gammadash1
    gammadash1_base = base_surface.gammadash1()
    gammadash1_offset = offset_surface.gammadash1()
    unitnormaldash1 = base_surface.unitnormaldash(1, 0)
    unitnormaldash2 = base_surface.unitnormaldash(0, 1)
    gammadash1_expected = gammadash1_base + offset_distance * unitnormaldash1
    
    max_diff = jnp.max(jnp.abs(gammadash1_offset - gammadash1_expected))
    print(f"  gammadash1 max difference: {max_diff}")
    assert jnp.allclose(gammadash1_offset, gammadash1_expected, atol=1e-10)
    print("✓ gammadash1() offset correctly")
    
    # Test gammadash2
    gammadash2_base = base_surface.gammadash2()
    gammadash2_offset = offset_surface.gammadash2()
    gammadash2_expected = gammadash2_base + offset_distance * unitnormaldash2
    
    max_diff = jnp.max(jnp.abs(gammadash2_offset - gammadash2_expected))
    print(f"  gammadash2 max difference: {max_diff}")
    assert jnp.allclose(gammadash2_offset, gammadash2_expected, atol=1e-10)
    print("✓ gammadash2() offset correctly")
    
    # Test area increases
    area_base = base_surface.area()
    area_offset = offset_surface.area()
    print(f"  Base area: {float(area_base):.6f}")
    print(f"  Offset area: {float(area_offset):.6f}")
    print(f"  Area ratio: {float(area_offset/area_base):.6f}")
    assert area_offset > area_base
    print("✓ Area increases with offset")
    
    # Test copy_and_set_quadpoints
    new_nphi = len(base_surface.quadpoints_phi) // 2
    new_ntheta = len(base_surface.quadpoints_theta) // 2
    new_quadpoints_phi = jnp.linspace(0, 1/base_surface.nfp, new_nphi, endpoint=False)
    new_quadpoints_theta = jnp.linspace(0, 1, new_ntheta, endpoint=False)
    
    offset_surface_new = offset_surface.copy_and_set_quadpoints(
        new_quadpoints_phi, new_quadpoints_theta
    )
    assert jnp.array_equal(offset_surface_new.quadpoints_phi, new_quadpoints_phi)
    assert jnp.array_equal(offset_surface_new.quadpoints_theta, new_quadpoints_theta)
    assert offset_surface_new.offset_distance == offset_distance
    print("✓ copy_and_set_quadpoints() works")
    
    # Test pytree
    from jax import tree_util
    children, aux = tree_util.tree_flatten(offset_surface)
    reconstructed = tree_util.tree_unflatten(aux, children)
    assert reconstructed.offset_distance == offset_surface.offset_distance
    assert jnp.allclose(reconstructed.gamma(), offset_surface.gamma())
    print("✓ Pytree flatten/unflatten works")
    
    print("\n✅ All tests passed with real surface data!")
    return True


def test_second_derivatives():
    """Test that second derivatives of unitnormal and offset surface work."""
    print("\n" + "=" * 60)
    print("Testing Second Derivatives")
    print("=" * 60)
    
    # Load test data
    _, _, _, _, qp = load_data()
    base_surface = qp.plasma_surface
    offset_distance = 0.1
    offset_surface = SurfaceOffsetJAX(base_surface, offset_distance)
    
    print("\n1. Testing unitnormaldash for second derivatives...")
    
    # Test that second derivatives can be computed
    try:
        d2_unitnormal_dphi2 = base_surface.unitnormaldash(2, 0)
        nphi, ntheta = d2_unitnormal_dphi2.shape[0], d2_unitnormal_dphi2.shape[1]
        print(f"  d²(unitnormal)/dphi² shape: {d2_unitnormal_dphi2.shape}")
        assert d2_unitnormal_dphi2.shape == (nphi, ntheta, 3)
        print("✓ unitnormaldash(2, 0) works")
        
        d2_unitnormal_dtheta2 = base_surface.unitnormaldash(0, 2)
        print(f"  d²(unitnormal)/dtheta² shape: {d2_unitnormal_dtheta2.shape}")
        assert d2_unitnormal_dtheta2.shape == (nphi, ntheta, 3)
        print("✓ unitnormaldash(0, 2) works")
        
        d2_unitnormal_mixed = base_surface.unitnormaldash(1, 1)
        print(f"  d²(unitnormal)/dphi dtheta shape: {d2_unitnormal_mixed.shape}")
        assert d2_unitnormal_mixed.shape == (nphi, ntheta, 3)
        print("✓ unitnormaldash(1, 1) works")
        
    except Exception as e:
        print(f"❌ Failed to compute second derivatives: {e}")
        raise
    
    print("\n2. Testing offset surface second derivatives...")
    
    # Test that offset surface second derivatives work
    try:
        d2_gamma_offset_dphi2 = offset_surface.gammadash(2, 0)
        print(f"  d²(gamma_offset)/dphi² shape: {d2_gamma_offset_dphi2.shape}")
        
        # Verify the formula: d²(gamma_offset)/dphi² = d²gamma/dphi² + d * d²(unitnormal)/dphi²
        expected = (base_surface.gammadash(2, 0) + 
                   offset_distance * base_surface.unitnormaldash(2, 0))
        max_diff = jnp.max(jnp.abs(d2_gamma_offset_dphi2 - expected))
        print(f"  Max difference from formula: {max_diff}")
        assert jnp.allclose(d2_gamma_offset_dphi2, expected, atol=1e-10)
        print("✓ offset_surface.gammadash(2, 0) works")
        
        d2_gamma_offset_dtheta2 = offset_surface.gammadash(0, 2)
        expected = (base_surface.gammadash(0, 2) + 
                   offset_distance * base_surface.unitnormaldash(0, 2))
        max_diff = jnp.max(jnp.abs(d2_gamma_offset_dtheta2 - expected))
        print(f"  Max difference from formula: {max_diff}")
        assert jnp.allclose(d2_gamma_offset_dtheta2, expected, atol=1e-10)
        print("✓ offset_surface.gammadash(0, 2) works")
        
        d2_gamma_offset_mixed = offset_surface.gammadash(1, 1)
        expected = (base_surface.gammadash(1, 1) + 
                   offset_distance * base_surface.unitnormaldash(1, 1))
        max_diff = jnp.max(jnp.abs(d2_gamma_offset_mixed - expected))
        print(f"  Max difference from formula: {max_diff}")
        assert jnp.allclose(d2_gamma_offset_mixed, expected, atol=1e-10)
        print("✓ offset_surface.gammadash(1, 1) works")
        
    except Exception as e:
        print(f"❌ Failed to compute offset surface second derivatives: {e}")
        raise
    
    print("\n3. Testing backward compatibility with legacy implementation...")
    
    # Compare autodiff vs legacy for first derivatives
    unitnormaldash1_new = base_surface.unitnormaldash(1, 0)
    unitnormaldash2_new = base_surface.unitnormaldash(0, 1)
    unitnormaldash1_legacy, unitnormaldash2_legacy = base_surface.unitnormaldash_legacy()
    
    max_diff_phi = jnp.max(jnp.abs(unitnormaldash1_new - unitnormaldash1_legacy))
    max_diff_theta = jnp.max(jnp.abs(unitnormaldash2_new - unitnormaldash2_legacy))
    
    print(f"  Max difference for d(unitnormal)/dphi: {max_diff_phi}")
    print(f"  Max difference for d(unitnormal)/dtheta: {max_diff_theta}")
    
    assert jnp.allclose(unitnormaldash1_new, unitnormaldash1_legacy, atol=1e-10)
    assert jnp.allclose(unitnormaldash2_new, unitnormaldash2_legacy, atol=1e-10)
    print("✓ Autodiff matches legacy implementation")
    
    print("\n✅ All second derivative tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SurfaceOffsetJAX with Real Test Data")
    print("=" * 60)
    
    try:
        test_with_real_surface()
        test_second_derivatives()
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
