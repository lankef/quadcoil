from quadcoil import quadcoil, SurfaceRZFourierJAX, get_quantity
import jax.numpy as jnp
from scipy.constants import mu_0


def generate_desc_scaling(objective_name, constraint_name, scales):
    '''
    A helper method for calculating the units of each objectives and constraints
    using quantities scales = desc.objectives.normalization.compute_scaling_factors(eq).
    The formula must be first implemented as quadcoil.objective.<quantity_name>_desc_unit.
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

# ----- An earlier, simpler wrapper. May be discontinued later. -----
def quadcoil_desc(
    desc_eq,
    vacuum:bool,
    plasma_M_theta:int,
    plasma_N_phi:int,
    desc_scaling:bool=True,
    **kwargs
):
    '''
    A simple DESC interface for ``quadcoil.quadcoil``, that replaces the following
    parameters using information extracted from DESC:
    
    .. code-block:: python

        nfp
        stellsym
        plasma_mpol
        plasma_ntor
        plasma_quadpoints_phi
        plasma_quadpoints_theta
        plasma_dofs
        net_poloidal_current_amperes
        Bnormal_plasma
        winding_dofs

    '''
    try:
        from desc.integrals import compute_B_plasma
        from desc.grid import LinearGrid
        from desc.objectives.normalization import compute_scaling_factors
    except:
        raise ModuleNotFoundError('DESC must be installed for the... DESC interface to run.')
    if 'plasma_coil_distance' in kwargs:
        # DESC surfaces have different handed-ness 
        # compared to simsopt surfaces. Because QUADCOIL
        # uses simsopt conventions, we flip the sign 
        # of offset direction here.
        kwargs['plasma_coil_distance'] = -kwargs['plasma_coil_distance'] 
    # We want to make sure DESC and QUADCOIL
    # uses the same grid on the plasma surface.
    # In case we want to use Bnormal_plasma 
    # from DESC.
    # To do this, we create a grid inside DESC.
    surface_grid = LinearGrid(
        NFP=desc_eq.NFP,
        # If we set this to sym it'll only evaluate 
        # theta from 0 to pi.
        sym=False, 
        M=plasma_M_theta,#Poloidal grid resolution.
        N=plasma_N_phi,
        rho=1.0
    )
    # Recovering quadpoints from the LinearGrid
    # for use in QUADCOIL and simsopt
    desc_plasma_quadpoints_theta = surface_grid.nodes[surface_grid.unique_theta_idx, 1]/jnp.pi/2
    desc_plasma_quadpoints_phi = surface_grid.nodes[surface_grid.unique_zeta_idx, 2]/jnp.pi/2
    desc_plasma_surf = SurfaceRZFourierJAX.from_desc(
        desc_eq.surface, 
        desc_plasma_quadpoints_phi,
        desc_plasma_quadpoints_theta
    )
    desc_net_poloidal_current_amperes = (
        -desc_eq.compute("G", grid=LinearGrid(rho=jnp.array(1.0)))["G"][0]
        / mu_0
        * 2
        * jnp.pi
    )
    # Calculating B_plasma if not vacuum
    if not vacuum:
        desc_Bnormal_plasma = compute_B_plasma(
            desc_eq, 
            eval_grid=surface_grid, 
            normal_only=True
            # chunk_size=B_plasma_chunk_size
        )
        # Reshape the output so that axis=0 is zeta (the toroidal coordinate)
        desc_Bnormal_plasma = surface_grid.meshgrid_reshape(desc_Bnormal_plasma, order='rzt')[0]
    else:
        desc_Bnormal_plasma = jnp.zeros((
            len(desc_plasma_quadpoints_phi),
            len(desc_plasma_quadpoints_theta)
        ))
    print('Finished reading DESC equilibrium. G =', desc_net_poloidal_current_amperes)
    print('Finished reading DESC equilibrium. G =', type(desc_net_poloidal_current_amperes))
    # Detect if the user has provided any arguments 
    # that will also-be auto-calculated using DESC. 
    # if any, these objectives will be discarded.
    redundant_arg_names = set(_DESC_DERIVED_ARGNAMES) & kwargs.keys()
    if redundant_arg_names:
        warnings.warn(
            'Redundant arguments detected: ' 
            + str(redundant_arg_names)
            + '. These arguments must be calculated from '\
            'the equilibrium. The provided values will be discarded.'
        )
    # Add all the equilibrium-dependent, non-differentiable args to quadcopil_args
    # Non-differentiable 
    # (This distinction is only for internal reference.
    # the desc _Objective wrapper of QUADCOIL uses a custom_jvp rule in 
    # quadcoil.io.jax, and that function makes the distinction between
    # differentiable and non-differentiable args.)
    kwargs['nfp'] = desc_eq.NFP
    kwargs['stellsym'] = desc_eq.sym
    kwargs['plasma_mpol'] = desc_plasma_surf.mpol
    kwargs['plasma_ntor'] = desc_plasma_surf.ntor
    kwargs['plasma_quadpoints_phi'] = desc_plasma_quadpoints_phi
    kwargs['plasma_quadpoints_theta'] = desc_plasma_quadpoints_theta
    # Differentiable 
    kwargs['plasma_dofs'] = desc_plasma_surf.dofs
    kwargs['net_poloidal_current_amperes'] = desc_net_poloidal_current_amperes
    kwargs['Bnormal_plasma'] = desc_Bnormal_plasma
    # For unconstrained cases, kwargs will not contain 
    # 'constraint_name'. But we still want to use the scaling helper function.
    # So we manually add a blank entry.
    if 'constraint_name' not in kwargs:
        kwargs['constraint_name'] = ()
    if desc_scaling:
        print('Overriding provided/default scaling with DESC scaling.')
        obj_unit_new, cons_unit_new = generate_desc_scaling(
            kwargs['objective_name'], 
            kwargs['constraint_name'], 
            compute_scaling_factors(desc_eq)
        )
        kwargs['objective_unit'] = obj_unit_new
        kwargs['constraint_unit'] = cons_unit_new
    return(
        quadcoil(**kwargs)
    )
