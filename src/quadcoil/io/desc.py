from quadcoil import SurfaceRZFourierJAX, get_quantity
import jax.numpy as jnp


def generate_desc_scaling(objective_name, constraint_name, scales):
    '''
    A helper method for calculating the units of each objectives and constraints
    using quantities scales = desc.objectives.normalization.compute_scaling_factors(eq).
    The formula must be first implemented as the ``desc_unit`` of ``quadcoil.quantities.<quantity_name>``.
    '''
    if isinstance(objective_name, str):
        objective_unit = get_quantity(objective_name).desc_unit(scales)
    else:
        objective_unit_new = []
        for obj_name in objective_name:
            objective_unit_new.append(get_quantity(obj_name).desc_unit(scales))
        objective_unit = tuple(objective_unit_new)
    
    constraint_unit_new = []
    for cons_name in constraint_name:
        constraint_unit_new.append(get_quantity(cons_name).desc_unit(scales))
    constraint_unit = tuple(constraint_unit_new)
    return(objective_unit, constraint_unit)

def load_FourierCurrentPotentialField(eq, scf, winding_quadpoints_phi=None, winding_quadpoints_theta=None, source_grid=None):
    try:
        from desc.vmec_utils import ptolemy_identity_rev
    except:
        raise ModuleNotFoundError(
            'DESC must be installed to load '
            'a FourierCurrentPotentialField.'
        )
    mm, nn, phis_raw, phic_raw = ptolemy_identity_rev(scf.Phi_basis.modes[:,1], scf.Phi_basis.modes[:,2], scf.Phi_mn)
    mpol = scf.M
    ntor = scf.N
    stellsym = scf.sym
    nfp = scf.NFP
    phic = phic_raw.flatten()
    phis = phis_raw.flatten()[1:]
    # When stellsym is enabled, Phi is a sin fourier series.
    # Otherwise, it's a sin-cos series.
    if stellsym:
        phi = jnp.concatenate([phis])
    else:
        phi = jnp.concatenate([phis, phic])
    if quadpoints_phi and quadpoints_theta:
        if source_grid:
            raise ValueError(
                'The user should provide either '
                '(quadpoints_phi, quadpoints_theta) or source_grid, '
                'not both.'
            )
        winding_quadpoints_phi_temp = winding_quadpoints_phi
        winding_quadpoints_theta_temp = winding_quadpoints_theta
    else:
        winding_quadpoints_phi_temp = jnp.linspace(0, 1, (source_grid.M*2+1) * nfp, endpoint=False)
        winding_quadpoints_theta_temp = jnp.linspace(0, 1, source_grid.N*2+1, endpoint=False)
    
    winding_surface = SurfaceRZFourierJAX.from_desc(
        scf_nescoil, 
        quadpoints_phi = winding_quadpoints_phi_temp,
        quadpoints_theta = winding_quadpoints_theta_temp,
    )
    return (qp, {'phi': phi})
