from quadcoil import QuadcoilParams, SurfaceRZFourierJAX
import jax.numpy as jnp
from simsopt import load
from simsopt.field import CurrentPotentialFourier, CurrentPotentialSolve

# Loading testing example
winding_surface, plasma_surface = load('surfaces.json')
cp = CurrentPotentialFourier(
    winding_surface, mpol=4, ntor=4,
    net_poloidal_current_amperes=11884578.094260072,
    net_toroidal_current_amperes=0,
    stellsym=True)
cp.set_dofs(jnp.array([  
    235217.63668779,  -700001.94517193,  1967024.36417348,
    -1454861.01406576, -1021274.81793687,  1657892.17597651,
    -784146.17389912,   136356.84602536,  -670034.60060171,
    194549.6432583 ,  1006169.72177152, -1677003.74430119,
    1750470.54137804,   471941.14387043, -1183493.44552104,
    1046707.62318593,  -334620.59690486,   658491.14959397,
    -1169799.54944824,  -724954.843765  ,  1143998.37816758,
    -2169655.54190455,  -106677.43308896,   761983.72021537,
    -986348.57384563,   532788.64040937,  -600463.7957275 ,
    1471477.22666607,  1009422.80860728, -2000273.40765417,
    2179458.3105468 ,   -55263.14222144,  -315581.96056445,
    587702.35409154,  -637943.82177418,   609495.69135857,
    -1050960.33686344,  -970819.1808181 ,  1467168.09965404,
    -198308.0580687 
]))
cpst = CurrentPotentialSolve(cp, plasma_surface, jnp.zeros(1024))
winding_surface_jax = SurfaceRZFourierJAX.from_simsopt(winding_surface)
plasma_surface_jax = SurfaceRZFourierJAX.from_simsopt(plasma_surface)
qp = QuadcoilParams(
    plasma_surface=plasma_surface_jax,
    winding_surface=winding_surface_jax,
    net_poloidal_current_amperes=cp.net_poloidal_current_amperes, 
    net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
    mpol=cp.mpol, 
    ntor=cp.ntor, 
)

def load_data():
    return(winding_surface, plasma_surface, cp, cpst, qp)


def compare(a, b):
    # print('Normalized error threhold: 1e-5')
    # print('Max error:', jnp.max(jnp.abs(a-b)), '; max |data|:', jnp.max(jnp.abs(a)))
    print('Absolute error:  ', jnp.max(jnp.abs(a-b)))
    print('Normalized error:', jnp.max(jnp.abs(a-b))/jnp.max(jnp.abs(a)))
    return(jnp.max(jnp.abs(a-b)) < jnp.max(jnp.abs(a))*1e-5)