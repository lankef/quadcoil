from jax import custom_jvp
from functools import partial
from quadcoil import quadcoil
import jax.numpy as jnp

def quadcoil_for_diff_full(
    plasma_dofs,
    net_poloidal_current_amperes:float,
    net_toroidal_current_amperes:float,
    Bnormal_plasma,
    plasma_coil_distance:float,
    winding_dofs,
    objective_weight,
    constraint_value,
    # bunch all of the non-differentiable inputs
    # together
    nondiff_args, 
):
    '''
    See ``quadcoil_for_diff``, except this function also
    outputs the ``QuadcoilParams``, ``phi_mn`` and ``status``
    from ``quadcoil.quadcoil``.

    Parameters
    ----------  
    plasma_dofs : ndarray
        (Static) The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    net_poloidal_current_amperes : float
        (Traced) The net poloidal current :math:`G`.
    net_toroidal_current_amperes : float, optional, default=0
        (Traced) The net toroidal current :math:`I`.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        (Traced) The magnetic field distribution on the plasma surface. Will be filled with zeros by default.
    plasma_coil_distance : float, optional, default=None
        (Traced) The coil-plasma distance. Is set to ``None`` by default, but a value must be provided if ``winding_dofs`` is not provided.
    winding_dofs : ndarray, shape (ndof_winding,)
        (Traced) The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    objective_weight : ndarray, optional, default=None
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    constraint_value : ndarray, optional, default=()
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    nondiff_args : dict, optional, default={}
        (Mixed) The rest of the parameters of ``quadcoil.quadcoil``.
        
    Returns
    -------
    out_dict_simple : dict
        A dictionary, where the keys are objective name and the values are their values.
    '''
    # Forcing value_only to True to save resources
    nondiff_args['value_only'] = True
    # We now convert into the a **kwargs for quadcoil. 
    # We first pop the positional, non-differentiable arguments
    # from the dict:
    quadcoil_kwargs = nondiff_args.copy()
    nfp = quadcoil_kwargs.pop('nfp')
    stellsym = quadcoil_kwargs.pop('stellsym')
    plasma_mpol = quadcoil_kwargs.pop('plasma_mpol')
    plasma_ntor = quadcoil_kwargs.pop('plasma_ntor')
    # We then insert the differentiable key word arguments
    quadcoil_kwargs['net_toroidal_current_amperes'] = net_toroidal_current_amperes
    quadcoil_kwargs['Bnormal_plasma'] = Bnormal_plasma
    quadcoil_kwargs['plasma_coil_distance'] = plasma_coil_distance
    quadcoil_kwargs['winding_dofs'] = winding_dofs
    quadcoil_kwargs['objective_weight'] = objective_weight
    quadcoil_kwargs['constraint_value'] = constraint_value
    out_dict, qp, cp_mn, solve_results = quadcoil(
        nfp, # positional, non-differentiable
        stellsym, # positional, non-differentiable
        plasma_mpol, # positional, non-differentiable
        plasma_ntor, # positional, differentiable
        plasma_dofs, # positional, differentiable
        net_poloidal_current_amperes, # positional, differentiable
        # non-diff args now contains all kwargs
        **quadcoil_kwargs
    )
    out_dict_simple = {}
    for key_i in out_dict.keys():
        out_dict_simple[key_i] = out_dict[key_i]['value']
        
    # The rest of the outputs are not differentiable
    # These outputs are currently commented out, but that may change 
    # after DESC ``_Objective`` can retain memory.
    return out_dict_simple, qp, cp_mn, solve_results

@partial(custom_jvp, nondiff_argnums=(8,))
def quadcoil_for_diff(
    plasma_dofs,
    net_poloidal_current_amperes:float,
    net_toroidal_current_amperes:float,
    Bnormal_plasma,
    plasma_coil_distance:float,
    winding_dofs,
    objective_weight,
    constraint_value,
    # bunch all of the non-differentiable inputs
    # together
    nondiff_args, 
):
    '''
    A wrapper for quadcoil that bundles all of the non-differentiable 
    parameters into the dict ``nondiff_args``, and evaluates only the 
    value of the metrics. This splits quadcoil into 2 parts: a jax 
    differentiable function ``quadcoil_for_diff`` and its ``custom_jvp``, 
    ``quadcoil_for_diff_jvp``, that stops auto-diff and replace the results
    with our own diff. This wrapper is necessary for the DESC interface, 
    because DESC automatically call autodiff on all objectives.

    Parameters
    ----------  
    plasma_dofs : ndarray
        (Static) The plasma surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    net_poloidal_current_amperes : float
        (Traced) The net poloidal current :math:`G`.
    net_toroidal_current_amperes : float, optional, default=0
        (Traced) The net toroidal current :math:`I`.
    Bnormal_plasma : ndarray, shape (nphi, ntheta), optional, default=None
        (Traced) The magnetic field distribution on the plasma surface. Will be filled with zeros by default.
    plasma_coil_distance : float, optional, default=None
        (Traced) The coil-plasma distance. Is set to ``None`` by default, but a value must be provided if ``winding_dofs`` is not provided.
    winding_dofs : ndarray, shape (ndof_winding,)
        (Traced) The winding surface degrees of freedom. Uses the ``simsopt.geo.SurfaceRZFourier.get_dofs()`` convention.
    objective_weight : ndarray, optional, default=None
        (Traced) The weights of the objective functions. Derivatives will be calculated w.r.t. this quantity.
    constraint_value : ndarray, optional, default=()
        (Traced) The constraint thresholds. Derivatives will be calculated w.r.t. this quantity.
    nondiff_args : dict, optional, default={}
        (Mixed) The rest of the parameters of ``quadcoil.quadcoil``.
        
    Returns
    -------
    out_dict_simple : dict
        A dictionary, where the keys are objective name and the values are their values.
    '''
    return quadcoil_for_diff_full(
        plasma_dofs,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        Bnormal_plasma,
        plasma_coil_distance,
        winding_dofs,
        objective_weight,
        constraint_value,
        # bunch all of the non-differentiable inputs
        # together
        nondiff_args, 
    )[0]

@quadcoil_for_diff.defjvp
def quadcoil_for_diff_jvp(nondiff_args, primals, tangents):
    '''
    The ``custom_jvp`` of ``quadcoil_for_diff``. 
    '''
    (
        plasma_dofs,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        Bnormal_plasma,
        plasma_coil_distance,
        winding_dofs,
        objective_weight,
        constraint_value,
    ) = primals

    (
        plasma_dofs_dot,
        net_poloidal_current_amperes_dot,
        net_toroidal_current_amperes_dot,
        Bnormal_plasma_dot,
        plasma_coil_distance_dot,
        winding_dofs_dot,
        objective_weight_dot,
        constraint_value_dot,
    ) = tangents
    nondiff_args = nondiff_args.copy()
    # Forcing value_only to False to enable gradients
    nondiff_args['value_only'] = False
    # We now construct convert nondiff_args into the kwargs of quadcoil. 
    # We first pop the positional, non-differentiable arguments
    # from the dict:
    nfp = nondiff_args.pop('nfp')
    stellsym = nondiff_args.pop('stellsym')
    plasma_mpol = nondiff_args.pop('plasma_mpol')
    plasma_ntor = nondiff_args.pop('plasma_ntor')
    # We then insert the differentiable key word arguments
    nondiff_args['net_toroidal_current_amperes'] = net_toroidal_current_amperes
    nondiff_args['Bnormal_plasma'] = Bnormal_plasma
    nondiff_args['plasma_coil_distance'] = plasma_coil_distance
    nondiff_args['winding_dofs'] = winding_dofs
    nondiff_args['objective_weight'] = objective_weight
    nondiff_args['constraint_value'] = constraint_value
    out_dict_full, qp, cp_mn, solve_results = quadcoil(
        nfp=nfp,
        stellsym=stellsym,
        plasma_mpol=plasma_mpol,
        plasma_ntor=plasma_ntor,
        plasma_dofs=plasma_dofs,
        net_poloidal_current_amperes=net_poloidal_current_amperes,
        # non-diff args now contains all kwargs
        **nondiff_args
    )
    # Recreate primal output:
    out_dict_primal = {}
    # Initialize tangent outputs
    out_dict_dot = {}
    for key_i in out_dict_full.keys():
        out_dict_primal[key_i] = out_dict_full[key_i]['value']
        # The shape of the second layer differs depending on the inputs. 
        # we handle them individually. 
        jvp_i = 0
        if plasma_dofs_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dplasma_dofs'] * plasma_dofs_dot)
        if net_poloidal_current_amperes_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dnet_poloidal_current_amperes'] * net_poloidal_current_amperes_dot)
        if net_toroidal_current_amperes_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dnet_toroidal_current_amperes'] * net_toroidal_current_amperes_dot)
        if Bnormal_plasma_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dBnormal_plasma'] * Bnormal_plasma_dot)
        if plasma_coil_distance_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dplasma_coil_distance'] * plasma_coil_distance_dot)
        if winding_dofs_dot is not None:
            jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dwinding_dofs'] * winding_dofs_dot)
        if objective_weight_dot is not None:
            # Converting an empty list/tuple to an array will produce a NaN.
            if len(objective_weight_dot) > 0:
                jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dobjective_weight'] * jnp.array(objective_weight_dot))
        if constraint_value_dot is not None:
            if len(constraint_value_dot) > 0:
                jvp_i += jnp.sum(out_dict_full[key_i]['grad']['df_dconstraint_value'] * jnp.array(constraint_value_dot))
        out_dict_dot[key_i] = jvp_i

    # The rest of the outputs are not differentiable
    # These outputs are currently commented out, but that may change 
    # after DESC ``_Objective`` can retain memory.
    # qp_dot = None
    # cp_mn_dot = None
    # solve_results_dot = None
    # return (out_dict_primal, qp, cp_mn, solve_results), (out_dict_dot, qp, cp_mn, solve_results)
    return (out_dict_primal), (out_dict_dot)