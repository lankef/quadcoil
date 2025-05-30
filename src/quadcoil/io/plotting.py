import numpy as np
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