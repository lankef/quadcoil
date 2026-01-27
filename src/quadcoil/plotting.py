from quadcoil import get_quantity
from quadcoil.wrapper import _parse_constraints
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def plot_quantity(
    name, qp, dofs,
    i_phi, j_phi,
    i_range=0.1, j_range=0.1,
    unit=1,
    ngrid=100,
    levels=30,
    constraint_mode=False,
    constraint_type=None,
    constraint_value=None,
    constraint_color=None,
    plot_contours=True
):
    """
    Make a 2D contour plot of f(qp, dofs) by varying phi[i] and phi[j].

    Parameters
    ----------
    name : string
        Name of a quantity.
    qp : object
        Problem object passed to f.
    dofs : dict
        Dictionary containing at least {"phi": np.ndarray}.
    i_phi, j_phi : int
        Indices into dofs["phi"] to vary.
    i_range, j_range : float
        Range to vary dofs["phi"] by.
    ngrid : int
        Number of points along each axis.
    levels : int
        Number of contour levels.
    constraint_mode : bool
        Whether the quantity is a constraint. If True, this function will mark constraint boundary.
    constraint_type : float
        Constraint type for the quantity.
    constraint_value : float
        Constraint threshold for the quantity.
    constraint_color : color,
        Color to plot the constraint threshold with.
    plot_contours : bool
        Whether to plot contours for the quantity. True by default. False is useful for overlaying multiple constraints. 
    """

    phi0 = dofs["phi"]
    phi_i0 = phi0[i_phi]
    phi_j0 = phi0[j_phi]

    phi_i_vals = np.linspace((1 - i_range) * phi_i0, (1 + i_range) * phi_i0, ngrid)
    phi_j_vals = np.linspace((1 - j_range) * phi_j0, (1 + j_range) * phi_j0, ngrid)

    if constraint_mode:
        g_ineq_list, h_eq_list, _ = _parse_constraints(
            constraint_name=(name,), 
            constraint_type=(constraint_type,), 
            constraint_unit=(unit,), 
            constraint_value=(constraint_value,), 
            smoothing='approx',
            smoothing_params={'lse_epsilon':1e-10},
        )
        if constraint_type == '=':
            f = h_eq_list[0]
        else:
            f = g_ineq_list[0]
    else:
        f = lambda qp, dofs, unit=unit: get_quantity(name)(qp, dofs)/unit

    def eval_f(phi_i, phi_j):
        phi_new = phi0.at[i_phi].set(phi_i).at[j_phi].set(phi_j)
        dofs_new = dict(dofs)
        dofs_new["phi"] = phi_new
        return f(qp, dofs_new)

    # Vectorize over the 2D grid
    vf = vmap(vmap(eval_f, in_axes=(None, 0)), in_axes=(0, None))
    if constraint_mode:
        if constraint_value is None: 
            raise AttributeError('constraint_value must be provided in constraint mode.')
    else:
        constraint_value=0
    Z = vf(phi_i_vals, phi_j_vals)

    # Convert to NumPy for plotting
    X, Y = np.meshgrid(phi_i_vals, phi_j_vals, indexing="ij")

    plt.scatter([phi_i0], [phi_j0], color='red', marker='x')
    if plot_contours:
        cs = plt.contour(
            np.asarray(X),
            np.asarray(Y),
            np.asarray(Z),
            levels=levels,
        )
        plt.colorbar(cs, label="f(qp, dofs)")
    if constraint_mode:
        # auto-assign colors for the constraint threshold line
        if not constraint_color:
            ax = plt.gca()
            constraint_color = ax._get_lines.get_next_color()
        plt.contour(X, Y, Z, levels=[0.0], colors=constraint_color, linewidths=2)
        # Shade regions where the constraint is violated            
        if np.max(Z) > 0:
            plt.contourf(
                X,
                Y,
                Z,
                levels=[0.0, np.max(Z)],
                colors=[constraint_color],
                alpha=0.2,
            )
        # add legend
        contour_proxy = plt.plot([], [], color=constraint_color, linewidth=2, label=name)
        plt.legend()
    plt.xlabel(f"phi[{i_phi}]")
    plt.ylabel(f"phi[{j_phi}]")
    plt.title(f"Contour of f varying phi[{i_phi}] and phi[{j_phi}]")
    plt.tight_layout()
    # plt.show()

def plot_quadcoil(
    qp, dofs,
    objective_name,
    objective_unit,
    constraint_name,
    constraint_type,
    constraint_unit,
    constraint_value,
    i_phi=0, j_phi=1,
    i_range=0.1, j_range=0.1,
    **kwargs
):
    plot_quantity(
        objective_name, qp, dofs,
        i_phi=i_phi, j_phi=j_phi,
        i_range=i_range, j_range=j_range,
        unit=objective_unit,
        constraint_mode=False,
    )
    if constraint_name:
        for i in range(len(constraint_name)):
            plot_quantity(
                constraint_name[i], qp, dofs,
                i_phi=i_phi, j_phi=j_phi,
                i_range=i_range, j_range=j_range,
                unit=constraint_unit[i],
                constraint_mode=True,
                constraint_type=constraint_type[i],
                constraint_value=constraint_value[i],
                plot_contours=False
            )
    plt.legend()
    plt.show()


def gamma_to_vtk(gamma, name):
    '''
    Converts a surface's quadrature points (produced by ``simsopt.Surface.gamma()``)
    into a vtk file.
    '''
    if name[-4:] != '.vts':
        name = name + '.vts'
    gamma = np.array(gamma)
    try:
        import pyvista as pv
    except:
        raise ImportError('pyvista must be installed to save vtk files.')
    # Assuming 'gamma' is your array of shape (N, M, 3) with [:,:,0] = r, [:,:,1] = phi, and [:,:,2] = z
    x = gamma[:, :, 0]
    y = gamma[:, :, 1]
    z = gamma[:, :, 2]

    # Flatten the arrays for grid points
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # Create the structured grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (x.shape[1], x.shape[0], 1)  # Set the grid dimensions, with 1 as the third dimension

    # Save as a VTK file
    grid.save(name)

def gamma_and_field_to_vtk(gamma, f, name):    
    '''
    Converts a surface's quadrature points (produced by ``simsopt.Surface.gamma()``)
    and a 3d vector field (xyz) into a vtk file.
    '''
    if name[-4:] != '.vts':
        name = name + '.vts'
    gamma = np.array(gamma)
    f = np.array(f)
    try:
        import pyvista as pv
    except:
        raise ImportError('pyvista must be installed to save vtk files.')
    # Assuming gamma and f are numpy arrays of shape (m, n, 3)
    m, n, _ = gamma.shape

    # Reshape gamma to a flat list of points
    points = gamma.reshape(-1, 3)

    # Reshape f to a flat list of vectors
    vectors = f.reshape(-1, 3)

    # Create a structured grid in PyVista
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [m, n, 1]  # Structured grid dimensions
    grid["f"] = vectors  # Add the vector field to the grid

    # Save the grid to a VTK file for ParaView
    grid.save(name)

def gamma_and_scalar_field_to_vtk(gamma, f_scalar, name):
    '''
    Converts a surface's quadrature points (produced by ``simsopt.Surface.gamma()``)
    and a scalar field (xyz) into a vtk file.
    '''
    if name[-4:] != '.vts':
        name = name + '.vts'
    gamma = np.array(gamma)
    f_scalar = np.array(f_scalar)
    try:
        import pyvista as pv
    except:
        raise ImportError('pyvista must be installed to save vtk files.')
    # Assuming `gamma` is your array of shape (N, M, 3) storing the x, y, z coordinates
    # and `f_scalar` is your array of shape (N, M) storing the scalar field
    x = gamma[:, :, 0]
    y = gamma[:, :, 1]
    z = gamma[:, :, 2]

    # Flatten the coordinate arrays and the scalar field for structured grid points
    points = np.c_[x.ravel(), y.ravel(), z.ravel()]
    scalars = f_scalar.ravel()

    # Create the structured grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (x.shape[1], x.shape[0], 1)  # Set dimensions with 1 in the third axis

    # Add scalar field as point data
    grid["Scalar Field"] = scalars

    # Save as a VTK file
    grid.save(name)
