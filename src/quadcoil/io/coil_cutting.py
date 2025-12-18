
# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
# Importing simsopt 
from quadcoil import QuadcoilParams
from quadcoil.quantity import Phi_with_net_current

def coil_zeta_theta_from_qp(
    qp:QuadcoilParams,
    dofs,
    coils_per_half_period=5,
    theta_shift=0): 
    stellsym = qp.stellsym
    nzeta_coil = len(qp.quadpoints_phi)
    nfp = qp.nfp 
    theta = qp.quadpoints_theta * 2 * np.pi 
    zeta = qp.quadpoints_phi * 2 * np.pi 
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes 

    # ------------------------
    # Load current potential
    # ------------------------
    current_potential = Phi_with_net_current(qp, dofs)

    # Now just generate a new monotonic array with the correct first value:
    theta = theta[0] + np.linspace(0,2*np.pi,len(theta),endpoint=False)

    d = 2*np.pi/nfp
    # We add a field period to the "left" and "right" of the 
    # present field period so that no contours are cutoff.
    zeta_3 = np.concatenate((zeta-d, zeta, zeta+d))
    # If the net current is non-neglegible, use the net current to calculate spacing.
    # Else, use the maximum phi to calculate spacing.
    # we normalize the current potential correspondingly into "data",
    # and also pad it to contain 3 field periods.
    if abs(net_poloidal_current_amperes) > np.finfo(float).eps:
        data = current_potential / net_poloidal_current_amperes * nfp  
        data_3 = np.concatenate((data-1,data,data+1))
    else:
        data = current_potential / np.max(current_potential)
        data_3 = np.concatenate((data,data,data))

    # We only generate contours based on the "middle"
    # field period. 
    d = 0.5/coils_per_half_period
    level_shift = theta_shift % 1 * d
    if stellsym:
        if theta_shift != 0:
            print('Warning: non-zero theta-shift is not permitted when stell-sym is true.')
        levels = np.linspace(0,0.5,coils_per_half_period,endpoint=False)
    else:
        levels = np.linspace(0,1,coils_per_half_period*2,endpoint=False) + level_shift
    # The contours are symmetric w.r.t. 2pi/nfp/2, i.e. stellarator
    # symmetric, by default.
    levels = levels + d/2
    fig = plt.figure()
    cdata = plt.contour(zeta_3, theta, np.transpose(data_3), levels, colors='k')
    plt.close(fig)
    num_coils_found = len(cdata.levels)
    if num_coils_found != 2*coils_per_half_period:
        print("Warning: The expected number of levels != 2 * coils_per_half_period.")

    contour_zeta=[]
    contour_theta=[]
    num_coils = 0
    for j in range(num_coils_found):
        paths_in_level = cdata.allsegs[j]
        # Each level can contain many paths.
        for p in paths_in_level:
            plt.plot(p[:, 0], p[:, 1])
            # These are the zeta (toroidal angles)
            # and theta of the paths.
            contour_zeta.append(p[:, 0])
            contour_theta.append(p[:, 1])
            num_coils += 1

    print('Cutting complete. Number of coils/half field period:', num_coils)
    return(contour_zeta, contour_theta)

# IFFT a array in real space to a sin/cos series used by sinsopt.geo.curve
def ifft_simsopt(x, order):
    assert len(x) >= 2*order  # the order of the fft is limited by the number of samples
    xf = rfft(x) / len(x)

    fft_0 = [xf[0].real]  # find the 0 order coefficient
    fft_cos = 2 * xf[1:order + 1].real  # find the cosine coefficients
    fft_sin = (-2 * xf[:order + 1].imag)[1:]  # find the sine coefficients
    dof = np.zeros(order*2+1)
    dof[0] = fft_0[0]
    dof[1::2] = fft_sin
    dof[2::2] = fft_cos

    return dof

# This script assumes the contours do not zig-zag back and forth across the theta=0 line,
# after shifting the current potential by theta_shift grid points.
# The nescin file is used to provide the coil winding surface, so make sure this is consistent with the regcoil run.
# ilambda is the index in the lambda scan which you want to select.
# def cut_coil(qp, qpst):
# filename = 'regcoil_out.li383.nc' # sys.argv[1]
# TODO: these tow have a lot of duplicate code. Clean up when finalizing how QUADCOIL is integrated.
def coil_xyz_from_qp(
    qp:QuadcoilParams,
    dofs,
    coils_per_half_period=1,
    theta_shift=0,
    save=False, save_name='placeholder'): 
    contour_zeta, contour_theta = coil_zeta_theta_from_qp(
        qp=qp,
        dofs=dofs,
        coils_per_half_period=coils_per_half_period,
        theta_shift=theta_shift
    )
    num_coils = len(contour_zeta)    
    nfp = qp.nfp 
    net_poloidal_current_amperes = qp.net_poloidal_current_amperes 

    # ------------------------
    # Load surface shape
    # ------------------------
    contour_R = []
    contour_Z = []
    for j in range(num_coils):
        contour_R.append(np.zeros_like(contour_zeta[j]))
        contour_Z.append(np.zeros_like(contour_zeta[j]))

    surf = qp.winding_surface.to_simsopt()
    for m in range(surf.mpol+1): # 0 to mpol
        for i in range(2*surf.ntor+1):
            n = i-surf.ntor
            crc = surf.rc[m, i]
            czs = surf.zs[m, i]
            if surf.stellsym:
                crs = 0
                czc = 0
            else: # Returns ValueError for stellsym cases
                crs = surf.get_rs(m, n)
                czc = surf.get_zc(m, n)
            for j in range(num_coils):
                angle = m*contour_theta[j] - n*contour_zeta[j]*surf.nfp
                # Was filled with zeroes.
                # Are lists because contou lengths are not uniform.
                contour_R[j] = contour_R[j] + crc*np.cos(angle) + crs*np.sin(angle)
                contour_Z[j] = contour_Z[j] + czs*np.sin(angle) + czc*np.cos(angle)

    contour_X = []
    contour_Y = []
    for j in range(num_coils):
        contour_X.append(contour_R[j]*np.cos(contour_zeta[j]))
        contour_Y.append(contour_R[j]*np.sin(contour_zeta[j]))
    coil_currents = net_poloidal_current_amperes / num_coils / qp.nfp
    if qp.stellsym:
        coil_currents = coil_currents/2
    if save: 
        coilsFilename='coils.'+save_name
        print("coilsFilename:",coilsFilename)
        # Write coils file
        f = open(coilsFilename,'w')
        f.write('periods '+str(nfp)+'\n')
        f.write('begin filament\n')
        f.write('mirror NIL\n')

        for j in range(num_coils):
            N = len(contour_X[j])
            for k in range(N):
                f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e}\n'.format(contour_X[j][k],contour_Y[j][k],contour_Z[j][k],coil_currents))
            # Close the loop
            k=0
            f.write('{:14.22e} {:14.22e} {:14.22e} {:14.22e} 1 Modular\n'.format(contour_X[j][k],contour_Y[j][k],contour_Z[j][k],0))

        f.write('end\n')
        f.close()
    return (
        contour_X,
        contour_Y,
        contour_Z,
        coil_currents,
        # np.sqrt(minSeparation2)
    )

# Load curves from lists of arrays containing x, y, and z.
def simsopt_curves_from_xyz(
    contour_X,
    contour_Y,
    contour_Z, 
    order=None, ppp=20):
    num_coils = len(contour_X)
    try:     
        from simsopt.geo import CurveXYZFourier
    except:
        raise ImportError('Simsopt is required to use the coil-cutting features.')
    # Calculating order
    if not order:
        order=float('inf')
        for i in range(num_coils):
            xArr = contour_X[i]
            yArr = contour_Y[i]
            zArr = contour_Z[i]
            for x in [xArr, yArr, zArr]:
                if len(x)//2<order:
                    order = len(x)//2
    
    coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
    # Compute the Fourier coefficients for each coil
    for ic in range(num_coils):
        xArr = contour_X[ic]
        yArr = contour_Y[ic]
        zArr = contour_Z[ic]

        # Compute the Fourier coefficients
        dofs=[]
        for x in [xArr, yArr, zArr]:
            dof_i = ifft_simsopt(x, order)
            dofs.append(dof_i)

        coils[ic].local_x = np.concatenate(dofs)
    return coils

def simsopt_coil_from_qp(
    qp, dofs, coils_per_half_period, theta_shift=0,
    method=coil_xyz_from_qp,
    base_mode=False,
    order=10, ppp=40):
    try:     
        from simsopt.field import Current # , Coil
        from simsopt.field.coil import coils_via_symmetries
    except:
        raise ImportError('Simsopt is required to use the coil-cutting features.')
    (
        contour_X,
        contour_Y,
        contour_Z,
        coil_currents,
        # min_separation
    ) = method(
        qp=qp,
        dofs=dofs,
        coils_per_half_period=coils_per_half_period,
        theta_shift=theta_shift,
        save=False
    )
    curves = simsopt_curves_from_xyz(
        contour_X,
        contour_Y,
        contour_Z, 
        order=order,
        ppp=ppp)
    currents=[]
    for i in range(len(curves)):
        currents.append(Current(coil_currents))
    if base_mode:
        return curves, currents
    coils = coils_via_symmetries(curves, currents, qp.nfp, qp.stellsym)
    return coils