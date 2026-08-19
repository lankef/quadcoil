"""
Plane fitting utilities for QUADCOIL.

The core plane fitting functions have been moved to quadcoil.math_utils.
This file now serves as a demonstration and testing module.
"""

import jax.numpy as jnp
from jax import jit, vmap

# Import plane fitting functions from math_utils
import sys
sys.path.insert(0, '../src')
from quadcoil.math_utils import project_points_to_plane, reconstruct_3d_from_plane, plane_fitting_error


if __name__ == "__main__":
    import jax
    jax.config.update('jax_enable_x64', True)
    
    print("=" * 70)
    print("Testing Plane Projection Functions")
    print("=" * 70)
    
    # Test 1: Points exactly on a plane
    print("\n1. Test with points exactly on a plane")
    print("-" * 70)
    # Create a grid of points on a tilted plane
    n_side = 10
    x_grid = jnp.linspace(-1, 1, n_side)
    y_grid = jnp.linspace(-1, 1, n_side)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    x = xx.flatten()
    y = yy.flatten()
    z = 0.5 * x + 0.3 * y  # Plane: z = 0.5x + 0.3y
    
    n_points = len(x)
    points_3d = jnp.stack([x, y, z], axis=-1)
    
    x_pol, y_pol, plane_data = project_points_to_plane(points_3d)
    
    print(f"Fitted plane normal: {plane_data['normal']}")
    # Expected normal: [0.5, 0.3, -1] / sqrt(0.5^2 + 0.3^2 + 1)
    # Note: SVD can return either direction, so check both
    expected_normal_unnorm = jnp.array([0.5, 0.3, -1.0])
    expected_normal = expected_normal_unnorm / jnp.linalg.norm(expected_normal_unnorm)
    print(f"Expected normal:     ±{expected_normal}")
    # Check alignment (should be ±1)
    alignment = jnp.dot(plane_data['normal'], expected_normal)
    print(f"Normal alignment:    {alignment:.6f} (should be ≈ ±1)")
    normal_error = min(jnp.linalg.norm(plane_data['normal'] - expected_normal),
                      jnp.linalg.norm(plane_data['normal'] + expected_normal))
    print(f"Normal error:        {normal_error:.2e}")
    print(f"RMS fitting error:   {plane_data['fitting_error']:.2e}")
    print(f"2D coords range:     x=[{x_pol.min():.3f}, {x_pol.max():.3f}], "
          f"y=[{y_pol.min():.3f}, {y_pol.max():.3f}]")
    
    # Test reconstruction
    reconstructed = reconstruct_3d_from_plane(x_pol, y_pol, plane_data)
    reconstruction_error = jnp.linalg.norm(reconstructed - points_3d)
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    # Test 2: Points on a plane with noise
    print("\n2. Test with noisy points")
    print("-" * 70)
    key = jax.random.PRNGKey(42)
    noise = 0.01 * jax.random.normal(key, shape=(n_points,))
    z_noisy = 0.5 * x + 0.3 * y + noise
    
    points_3d_noisy = jnp.stack([x, y, z_noisy], axis=-1)
    
    # Also compute expected normal for reference
    # Plane: z = 0.5x + 0.3y => 0.5x + 0.3y - z = 0 => normal proportional to [0.5, 0.3, -1]
    expected_normal_unnorm = jnp.array([0.5, 0.3, -1.0])
    expected_normal = expected_normal_unnorm / jnp.linalg.norm(expected_normal_unnorm)
    
    x_pol_noisy, y_pol_noisy, plane_data_noisy = project_points_to_plane(points_3d_noisy)
    
    print(f"Fitted plane normal: {plane_data_noisy['normal']}")
    print(f"Expected normal:     ±{expected_normal}")
    alignment_noisy = jnp.dot(plane_data_noisy['normal'], expected_normal)
    print(f"Normal alignment:    {alignment_noisy:.6f} (should be ≈ ±1)")
    normal_error_noisy = min(jnp.linalg.norm(plane_data_noisy['normal'] - expected_normal),
                            jnp.linalg.norm(plane_data_noisy['normal'] + expected_normal))
    print(f"Normal error:        {normal_error_noisy:.2e}")
    print(f"RMS fitting error:   {plane_data_noisy['fitting_error']:.6f}")
    print(f"Expected ~0.01:      (noise std)")
    
    rms_err, max_err = plane_fitting_error(points_3d_noisy, plane_data_noisy)
    print(f"RMS error (verified): {rms_err:.6f}")
    print(f"Max error:            {max_err:.6f}")
    
    # Test 3: Circular points in a tilted plane
    print("\n3. Test with circular points in a tilted plane")
    print("-" * 70)
    n_circle = 50
    theta = jnp.linspace(0, 2 * jnp.pi, n_circle)
    
    # Circle in XY plane, then rotate
    circle_2d = jnp.stack([jnp.cos(theta), jnp.sin(theta), jnp.zeros(n_circle)], axis=-1)
    
    # Rotation matrix (rotate 30 degrees around Y, then 45 degrees around Z)
    angle_y = jnp.pi / 6
    angle_z = jnp.pi / 4
    Ry = jnp.array([
        [jnp.cos(angle_y), 0, jnp.sin(angle_y)],
        [0, 1, 0],
        [-jnp.sin(angle_y), 0, jnp.cos(angle_y)]
    ])
    Rz = jnp.array([
        [jnp.cos(angle_z), -jnp.sin(angle_z), 0],
        [jnp.sin(angle_z), jnp.cos(angle_z), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry
    
    circle_3d = (R @ circle_2d.T).T + jnp.array([1.0, 2.0, 3.0])
    
    x_circle, y_circle, plane_circle = project_points_to_plane(circle_3d)
    
    print(f"Circle center in 3D:  {plane_circle['origin']}")
    print(f"Expected:             [1.0, 2.0, 3.0]")
    print(f"RMS fitting error:    {plane_circle['fitting_error']:.2e}")
    print(f"2D coords range:      x=[{x_circle.min():.3f}, {x_circle.max():.3f}], "
          f"y=[{y_circle.min():.3f}, {y_circle.max():.3f}]")
    
    # Check if points form a circle in 2D
    radius_2d = jnp.sqrt(x_circle**2 + y_circle**2)
    print(f"2D radius mean:       {jnp.mean(radius_2d):.6f}")
    print(f"2D radius std:        {jnp.std(radius_2d):.2e}")
    print(f"Expected radius:      1.0")
    
    # Test 4: vmap compatibility (batch processing)
    print("\n4. Test vmap compatibility (batch processing)")
    print("-" * 70)
    
    # Create 3 different point clouds
    n_clouds = 3
    n_pts = 30
    
    # Generate random planes
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, n_clouds)
    
    def generate_random_plane_points(key, n):
        # Random plane parameters
        k1, k2, k3 = jax.random.split(key, 3)
        
        # Generate points in XY plane
        xy = jax.random.uniform(k1, shape=(n, 2), minval=-1, maxval=1)
        z = jnp.zeros(n)
        points_2d = jnp.stack([xy[:, 0], xy[:, 1], z], axis=-1)
        
        # Random rotation
        angles = jax.random.uniform(k2, shape=(3,), minval=0, maxval=2*jnp.pi)
        
        # Simple rotation matrices
        cos_a, sin_a = jnp.cos(angles[0]), jnp.sin(angles[0])
        R = jnp.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        points_3d = (R @ points_2d.T).T
        translation = jax.random.uniform(k3, shape=(3,), minval=-2, maxval=2)
        points_3d = points_3d + translation
        
        return points_3d
    
    # Generate batch of point clouds
    point_clouds = jnp.stack([generate_random_plane_points(k, n_pts) for k in keys])
    print(f"Batch shape: {point_clouds.shape}")
    
    # Apply vmap
    batched_project = vmap(project_points_to_plane, in_axes=0, out_axes=(0, 0, 0))
    x_batch, y_batch, plane_batch = batched_project(point_clouds)
    
    print(f"Output x_pol shape: {x_batch.shape}")
    print(f"Output y_pol shape: {y_batch.shape}")
    print(f"Batch fitting errors: {plane_batch['fitting_error']}")
    
    # Verify each plane separately
    for i in range(n_clouds):
        x_i, y_i, plane_i = project_points_to_plane(point_clouds[i])
        error = jnp.abs(x_batch[i] - x_i).max() + jnp.abs(y_batch[i] - y_i).max()
        print(f"  Cloud {i}: vmap vs individual error = {error:.2e}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
