import numpy as np
from quadcoil import SurfaceRZFourierJAX
from desc.geometry.surface import FourierRZToroidalSurface
def save_focus(surface, filename):
    '''
    Saves a Simsopt `SurfaceRZFourier`, a DESC `FourierRZToroidalSurface`,
    or a QUADCOIL `SurfaceRZFourierJAX`
    to a `.plasma` file used for FOCUS or FAMUS under 
    `CASE_SURFACE=0`.

    Parameters
    ----------
    surface
        The surface tp save.
    filename
        The filename to save.
    '''
    if isinstance(surface, FourierRZToroidalSurface):
        surface = SurfaceRZFourierJAX.from_desc(
            surface,
            quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 32, endpoint=False)
        )
    if isinstance(surface, SurfaceRZFourierJAX):
        surface = surface.to_simsopt()
    replace_err = lambda x: 0 if isinstance(x, ValueError) or not np.isfinite(x) else x
    lines_to_write = [
        '# N_modes N_fp  N_bn\n',
        str(len(surface.get_dofs())) + '  ' + str(surface.nfp) + '  ' + '0\n', # No Bnorm modes  
        '#------- plasma boundary------\n',
        '#  n   m   Rbc   Rbs    Zbc   Zbs\n',
    ]
    for m in np.arange(0, surface.mpol+1):
        for n in np.arange(-surface.ntor, surface.ntor+1):
            Rbc = surface.get_rc(m, n)
            Rbs = surface.get_rs(m, n)
            Zbc = surface.get_zc(m, n)
            Zbs = surface.get_zs(m, n)
            lines_to_write+=[
                str(n) + '  '
                + str(m) + '  '
                + str(replace_err(Rbc)) + '  '
                + str(replace_err(Rbs)) + '  '
                + str(replace_err(Zbc)) + '  '
                + str(replace_err(Zbs)) + '\n'
            ]
    if not filename.endswith(".plasma"):
        filename += ".plasma"
    with open(filename, "w") as f:
        f.writelines(lines_to_write)