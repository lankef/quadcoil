#!/usr/bin/env python3
"""
Example demonstrating SurfaceOffsetJAX usage.

Run from the tests/ directory:
    cd tests && python ../example_offset_surface.py
"""

import jax.numpy as jnp
from quadcoil.surface import SurfaceOffsetJAX
from load_test_data import load_data

# Load a real plasma surface
_, _, _, _, qp = load_data()
plasma_surface = qp.plasma_surface

print("=" * 60)
print("SurfaceOffsetJAX Usage Example")
print("=" * 60)

# Example 1: Create a simple offset surface
print("\n1. Creating offset surface:")
print(f"   Base surface: nfp={plasma_surface.nfp}, {len(plasma_surface.quadpoints_phi)}×{len(plasma_surface.quadpoints_theta)} grid")

offset_distance = 0.15  # 15cm offset
winding_surface = SurfaceOffsetJAX(plasma_surface, offset_distance)
print(f"   Offset distance: {offset_distance}m")
print(f"   ✓ Created winding surface")

# Example 2: Use all standard surface methods
print("\n2. Using surface methods (all work automatically):")
gamma = winding_surface.gamma()
print(f"   gamma(): {gamma.shape}")

normal = winding_surface.normal()
print(f"   normal(): {normal.shape}")

area = winding_surface.area()
print(f"   area(): {float(area):.4f} m²")

da = winding_surface.da()
print(f"   da(): {da.shape}")

# Example 3: Compare with base surface
print("\n3. Comparing base vs offset surface:")
base_area = plasma_surface.area()
offset_area = winding_surface.area()
print(f"   Base area:   {float(base_area):.4f} m²")
print(f"   Offset area: {float(offset_area):.4f} m²")
print(f"   Ratio:       {float(offset_area/base_area):.4f}")

# Example 4: Use with copy_and_set_quadpoints
print("\n4. Changing quadrature resolution:")
new_quadpoints_phi = jnp.linspace(0, 1/plasma_surface.nfp, 64, endpoint=False)
new_quadpoints_theta = jnp.linspace(0, 1, 64, endpoint=False)
winding_surface_hires = winding_surface.copy_and_set_quadpoints(
    new_quadpoints_phi, new_quadpoints_theta
)
print(f"   Original: {len(winding_surface.quadpoints_phi)}×{len(winding_surface.quadpoints_theta)}")
print(f"   New:      {len(winding_surface_hires.quadpoints_phi)}×{len(winding_surface_hires.quadpoints_theta)}")
print(f"   ✓ Offset distance preserved: {winding_surface_hires.offset_distance}m")

# Example 5: Use in QuadcoilParams
print("\n5. Using with QuadcoilParams:")
print(f"   Can pass offset surface as eval_surface:")
print(f"   qp = QuadcoilParams(")
print(f"       eval_surface=SurfaceOffsetJAX(plasma_surf, 0.2),")
print(f"       ...)")
print(f"   Then _K(), normal(), da() etc. automatically use offset surface")

# Example 6: Different offset distances
print("\n6. Creating multiple winding surfaces:")
for d in [0.05, 0.10, 0.15, 0.20]:
    ws = SurfaceOffsetJAX(plasma_surface, d)
    area = ws.area()
    print(f"   d={d:.2f}m → area={float(area):.4f}m² (ratio: {float(area/base_area):.3f})")

print("\n" + "=" * 60)
print("✅ All operations completed successfully!")
print("=" * 60)
print("\nKey points:")
print("  • SurfaceOffsetJAX wraps any SurfaceJAX subclass")
print("  • All surface methods work automatically")
print("  • Fully compatible with JAX transformations (jit, grad, vmap)")
print("  • Use anywhere a SurfaceJAX is expected")
print("  • No need to pass offset parameters through functions")
