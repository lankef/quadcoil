from jax import custom_jvp
from functools import partial
from quadcoil import quadcoil
import jax.numpy as jnp
import warnings
import jax
# A list of differentiable arguments of QUADCOIL.
# Will be ignored from the kwargs of gen_quadcoil_for_diff
QUADCOIL_DIFF_ARGS = [
    'plasma_dofs',
    'net_poloidal_current_amperes',
    'net_toroidal_current_amperes',
    'Bnormal_plasma',
    'plasma_coil_distance',
    'winding_dofs',
    'objective_weight',
    'constraint_value',
]

def gen_quadcoil_for_diff(**kwargs):
    # Generate a quadcoil call taking only:
    # plasma_dofs,
    # net_poloidal_current_amperes:float,
    # net_toroidal_current_amperes:float,
    # Bnormal_plasma,
    # plasma_coil_distance:float,
    # winding_dofs,
    # objective_weight,
    # constraint_value,
    # And its custom_jvp using partal.

    # Copy the kwargs, remove the variables that are differentiable
    partial_kwargs = {}
    for key in kwargs.keys():
        if key in QUADCOIL_DIFF_ARGS:
            warnings.warn(key + " found in kwargs. This '\
            'is a differentiable argument and will be overridden.")
        else:
            partial_kwargs[key] = kwargs[key]
    # A partial of quadcoil taking only the 
    # differentiable arguments, and does not 
    # output derivatives
    quadcoil_temp = partial(
        quadcoil, 
        value_only=True,
        **partial_kwargs
    )    
    # A partial of quadcoil taking only the 
    # differentiable arguments, and preserves
    # the full output.
    quadcoil_full = partial(
        quadcoil, 
        value_only=False,
        **partial_kwargs
    )
    # A partial of quadcoil taking only the 
    # differentiable arguments, and outputs
    # value in a simpler dict.
    @partial(custom_jvp)
    def quadcoil_for_diff(
        plasma_dofs,
        net_poloidal_current_amperes,
        net_toroidal_current_amperes,
        Bnormal_plasma,
        plasma_coil_distance,
        winding_dofs,
        objective_weight,
        constraint_value,
    ):
        out_dict, qp, cp_mn, solve_results = quadcoil_temp(
            plasma_dofs=plasma_dofs,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal_plasma,
            plasma_coil_distance=plasma_coil_distance,
            winding_dofs=winding_dofs,
            objective_weight=objective_weight,
            constraint_value=constraint_value,
        )
        out_dict_simple = {}
        for key_i in out_dict.keys():
            out_dict_simple[key_i] = out_dict[key_i]['value']
            
        # The rest of the outputs are not differentiable
        # These outputs are currently commented out, but that may change 
        # after DESC ``_Objective`` can retain memory.
        return out_dict_simple

    @quadcoil_for_diff.defjvp
    def quadcoil_for_diff_jvp(primals, tangents):
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

        out_dict_full, qp, cp_mn, solve_results = quadcoil_full(
            plasma_dofs=plasma_dofs,
            net_poloidal_current_amperes=net_poloidal_current_amperes,
            net_toroidal_current_amperes=net_toroidal_current_amperes,
            Bnormal_plasma=Bnormal_plasma,
            plasma_coil_distance=plasma_coil_distance,
            winding_dofs=winding_dofs,
            objective_weight=objective_weight,
            constraint_value=constraint_value,
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
                if not jnp.isscalar(objective_weight_dot):
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
    return quadcoil_full, quadcoil_for_diff