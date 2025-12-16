def quadcoil_to_simsopt_cp(qp, dofs):
    try:
        from simsopt.field import CurrentPotentialFourier
    except:
        raise ImportError(
            'Analyzing the winding surface result in simsopt requires '
            'a branch with CurrentPotentialFourier and WindingSurfaceField '
            '(as of 2025 this is in the `regcoil` branch.)')
    cp_out = CurrentPotentialFourier(
        winding_surface=qp.winding_surface.to_simsopt(), 
        net_poloidal_current_amperes=qp.net_poloidal_current_amperes,
        net_toroidal_current_amperes=0, 
        nfp=qp.nfp, 
        stellsym=qp.stellsym,
        mpol=qp.mpol, ntor=qp.ntor,
    )
    cp_out.set_dofs(dofs['phi'])
    return cp_out