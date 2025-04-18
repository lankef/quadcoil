from typing import Callable, List, Any

class _Quantity: 
    r'''
    Interior point methods require :math:`C^1` objectives. Therefore, 
    QUADCOIL has to convert non-:math:`C^1`, convex objectives, such as 
    L-:math:`\infty` or L1 norms, into a combination of :math:`C^1` objectives, constraints
    and auxillary variables. For example, the L-:math:`\infty` norm 
    :math:`\max_{ws}\Phi`, is actually:
    
    .. math::
   
        \max_{ws}\Phi = \min s, \text{ where } |\Phi|\leq s.

    This "under-the-hood" implementation is less intuitive, and will not work for the optimum
    of a different problem that does not contain the same auxillary variable. To make 
    QUADCOIL (hopefully) easier to use and develop for, we've came up with the ``_Quantity`` class.

    ``_Quantity`` is designed as a class for storing all these different forms of the same
    physical quantity in the same place. You can think of it as a ``FunctionType`` that 
    switches between: 
        * A "surface-level", non-:math:`C^1` implementation that's 
        easy to understand and works for all problems. This is ``eff_val_func``.
        * An "under-the-hood" implementation :math:`C^1`, but nay not be well-defined in
        another problem with different auxillary vars, and is less intuitive to the user.
        These are ``val_func, aux_g_ineq_func, aux_h_eq_func, aux_dofs_init``.

    To add a new quantity to QUADCOIL, a developer only need to create a new **instance** (not subclass!)
    of ``_Quantity`` and provide all the functions and limitations necessary. Users will import and use
    these instances like functions, and why'll work across all cases, and developers can use the internal checks in 
    ``_Quantity`` to help make sure that they've implemented things properly.

    To developers familiar with DESC, this kind of design may seem somewhat similar to
    ``_Objective``, but again, please think of instances of ``_Quantity`` as 
    functions rather than objects. It stores nether the problem setup, nor the state during a solve.
    It's purely for use by ``quadcoil/wrapper.py`` to construct the :math:`f, g, h` of the constrained 
    optimization problem.

    Quantities with auxillary variables are often incompatible with ``'=='`` and ``'>='`` 
    constraints. For example, :math:`\max_{ws}\Phi=\alpha` and :math:`\max_{ws}\Phi\geq\alpha`
    are both non-convex constraints, and not supported by QUADCOIL. The compatibility is 
    stored in ``Objective``. 
    
    Parameters
    ----------
    val_func : Callable(qp: QuadcoilParams, dofs: dict)
        The value of the quantity. Must be :math:`C^1`.
    eff_val_func : Callable(qp: QuadcoilParams, dofs: dict)
        A "real" implementation of the quantity that can directly be calculated from :math:`\Phi_{sv}`
        without using auxillary variables. Can be :math:`C^0` rather than :math:`C^1`. This is the 
        "surface-level" imnplementation that works across different problems, and that 
        QUADCOIL users will typically access.
    aux_g_ineq_func : Callable(qp: QuadcoilParams, dofs: dict)
        The auxillary inequality constraints required by the objective, in the form 
        of :math:`g(x)\leq0`.  
    val_unit_to_g_ineq_unit : Callable(qp: QuadcoilParams, f_unit: float)
        The auxillary constraint of the quantity often have a different unit from the 
        value of the quantity itself. For example, the L-1 norm of a field :math:`v` has the unit of 
        :math:`(\text{unit}_v) m^2`, whereas its auxillary constraints have the unit of
        :math:`(\text{unit}_v)`. To allow the user to appoint one unit for the entire quantity, 
        we must provide a unit conversion function. The function will access the problem setup (but not state)
        via a ``QuadcoilParams``.
    aux_h_eq_func : Callable(qp: QuadcoilParams, dofs: dict)
        The auxillary equality constraints required by the objective, in the form 
        of :math:`h(x)=0`.  
    val_unit_to_h_eq_unit : Callable(qp: QuadcoilParams, f_unit: float)
        The unit conversion function for ``aux_h_eq_func``.
    aux_dofs_init : dict{str: scalar, array or Callable(qp: QuadcoilParams)}
        The names of the auxillary variables and callables for evaluating their initial values
        in terms of dofs.
    compatibility : List[str]
        Whether this quantity can serve as an objective, or appear in
        ``'<='``  ``'=='`` and (or) ``'>='`` constraints. Must be a ``List`` containing one or 
        more of ``'f'``, ``'<='``  ``'=='`` and (or) ``'>='``.
    desc_unit : Callable(scale: dict)
        For calculating the unit of this quantity in DESC.

    Attributes
    ----------
    val_func : Callable(x: dict)
    eff_val_func : Callable(x: dict)
    aux_g_ineq_func : Callable(x: dict)
    aux_h_eq_func : Callable(x: dict)
    aux_dofs_init : dict{str: Callable(QuadcoilParams) -> Tuple}
    desc_unit : Callable(scale: dict)
    compatibility : List[str]
    '''

    _used_aux_names = set()  # class-level registry for checking duplicate aux var names

    def __init__(
        self, 
        val_func, eff_val_func, 
        aux_g_ineq_func,
        val_unit_to_g_ineq_unit,
        aux_h_eq_func,
        val_unit_to_h_eq_unit,
        aux_dofs_init, 
        compatibility, 
        desc_unit
    ):
        if isinstance(val_func, _Quantity):
            raise TyperError('val_func must not be a _Quantity.')
        if isinstance(eff_val_func, _Quantity):
            raise TyperError('eff_val_func must not be a _Quantity.')
        if isinstance(aux_g_ineq_func, _Quantity):
            raise TyperError('aux_g_ineq_func must not be a _Quantity.')
        if isinstance(aux_h_eq_func, _Quantity):
            raise TyperError('aux_h_eq_func must not be a _Quantity.')
        # Checking whether the auxillary variable name has already been used elsewhere.
        if aux_dofs_init is not None:
            if not isinstance(aux_dofs_init, dict):
                raise TypeError('aux_dofs_init muse be a dict.')
            aux_names = set(aux_dofs_init.keys())
            dup_aux_names = self.__class__._used_aux_names.intersection(aux_names)
            # Check for duplicates
            if dup_aux_names:
                raise ValueError(
                    f'The auxillary variable name {dup_aux_names} '\
                    'has already been used. Please choose a different one.'\
                    'If you are not developing new quantities, this is an issue with '\
                    'the implementation of an existing quantity. Please contact the developers.'
                )
            # Check for typos in compatibility
            for item in compatibility:
                if item not in ['<=', '==', '>=', 'f']:
                    raise ValueError(f'`compatibility` contains illegal value, {item}')
            self.__class__._used_aux_names = self.__class__._used_aux_names | aux_names
        # Setting attributes
        self.val_func = val_func
        self.eff_val_func = eff_val_func
        self.aux_g_ineq_func = aux_g_ineq_func
        self.aux_h_eq_func = aux_h_eq_func
        self.aux_dofs_init = aux_dofs_init
        self.compatibility = compatibility
        self.desc_unit = desc_unit
        self.val_unit_to_g_ineq_unit = val_unit_to_g_ineq_unit
        self.val_unit_to_h_eq_unit = val_unit_to_h_eq_unit
        

    def __call__(self, qp, dofs):
        '''
        Evaluate this quantity. For  convenience.
        
        Parameters
        ----------
        qp : QuadcoilParams
            (static) The problem setup, including the plasma surface, 
            the winding surface, the currents, but not the objective/constraint choices.
        dofs : dict
            (static) A dictionary storing the degrees of freedom in a QUADCOIL problem. 
        '''
        return self.eff_val_func(qp, {'phi': dofs['phi']})
    
    def generate_linf_norm(func, aux_argname, desc_unit, positive_definite=False):
        '''
        Generates the auxillary constraints for an L-:math:`\infty`
        norm. See documentations for ``quadcoil.objective.Objective``.
        An L-:math:`\infty` norm term in an objective or :math:`\leq`
        constraint can be represented by:

        .. math::

            max|f| &= p,\\
            \text{where}&\\
             f - p &\leq 0\\
            -f - p &\leq 0
        
        This is a scalar, convex quantity that's only compatible as an 
        objective term or with ``'<='`` constraints.

        Parameters
        ----------  
        func : Callable
            The source function :math:`f` to convert into an L-:math:`\infty` norm.
        aux_argname : str
            The name of the auxillary variable. Must be unique among all supported
            ``_Quantity``s.
        maxdesc_unit : Callable
            A callable calculating the quantity's unit in DESC.
        positive_definite : bool, optional, default=False
            Whether ``func`` is positive definite. If ``True``, then
            the second constraint is not required.

        Returns
        -------
        A ``_Quantity``.
        '''
        # The objective/constraint form of this L-infinity
        # norm is just a function that returns the auxillary variable.
        val_func = lambda qp, dofs, aux_argname=aux_argname: dofs[aux_argname]
        # The effective function
        eff_val_func = lambda qp, dofs, func=func: jnp.max(jnp.abs(func(qp, dofs)))
        # The constraints
        def aux_g_ineq_func(qp, dofs, func=func, aux_argname=aux_argname):
            field = func(qp, dofs)
            p_aux = dofs[aux_argname]
            g_plus  =  field - p_aux #  f - p <=0
            # We need only half of the constraints if f is positive definite
            if positive_definite:
                return g_plus
            g_minus = -field - p_aux # -f - p <=0
            return jnp.concatenate([
                jnp.atleast_1d(g_plus),
                jnp.atleast_1d(g_minus),
            ])
        
        return _Quantity(
            val_func=val_func,
            eff_val_func=eff_val_func, 
            aux_g_ineq_func=aux_g_ineq_func,
            # The auxillary constraint g of L-inf norms
            # has the same unit as its value.
            val_unit_to_g_ineq_unit=lambda qp, unit: unit,  
            aux_h_eq_func=None, 
            val_unit_to_h_eq_unit=None,
            aux_dofs_init={aux_argname: eff_val_func},
            compatibility=['f', '<='], 
            desc_unit=desc_unit,
        )

    def generate_l1_norm(func, aux_argname, desc_unit, positive_definite=False):
        '''
        Generates the auxillary constraints for an L-1 
        norm. See documentations for ``quadcoil.objective.Objective``.
        An L-1 norm in an objective or a :math:`\leq` constraint is equivalent to:

        .. math::

            \|f\|_1 = \int da |f| &= \sum_{i=1}^{i=N_{grid}} N p_i,\\
            \text{where}\\
            N \text{ is the length of the surface normal}\\
             f_i - p_i &\leq 0\\
            -f_i - p_i &\leq 0
        
        This is a scalar, convex quantity that's only compatible as an 
        objective term or with ``'<='`` constraints.

        Parameters
        ----------  
        func : Callable
            The source function :math:`f` to convert into an L-:math:`\infty` norm.
        func_shape : Tuple
            The shape of ``func``'s output. Muse be 
        aux_argname : str
            The name of the auxillary variable. Must be unique among all supported
            ``_Quantity``s.
        maxdesc_unit : Callable
            A callable calculating the quantity's unit in DESC.
        positive_definite : bool, optional, default=False
            Whether ``func`` is positive definite. If ``True``, then
            the second constraint is not required.

        Returns
        -------
        A ``_Quantity``.
        '''
        # The objective/constraint form of this L-1
        # norm is the surface integral pf the aux variable.
        def val_func(qp, dofs, aux_argname=aux_argname):
            return qp.eval_surface.integrate(dofs[aux_argname])*qp.nfp
        # The effective function
        def eff_val_func(qp, dofs, func=func):
            return qp.eval_surface.integrate(jnp.abs(
                func(qp, dofs)
            ))*qp.nfp
        # The constraints
        def aux_g_ineq_func(qp, dofs, func=func, aux_argname=aux_argname):
            field = func(qp, dofs)
            p_aux = dofs[aux_argname]
            g_plus  =  field - p_aux #  f - p <=0
            # We need only half of the constraints if f is positive definite
            if positive_definite:
                return g_plus
            g_minus = -field - p_aux # -f - p <=0
            return jnp.concatenate([
                jnp.atleast_1d(g_plus),
                jnp.atleast_1d(g_minus),
            ])
        # The unit of L-1 norm is m^2 (unit),
        # but the unit of its auxillary constraints
        # are (unit). Therefore, g_unit is val_unit \
        # divided by the surface's area.
        def val_unit_to_g_ineq_unit(qp, unit):
            return unit / qp.eval_surface.area()*qp.nfp
        return _Quantity(
            val_func=val_func,
            eff_val_func=eff_val_func, 
            aux_g_ineq_func=aux_g_ineq_func,
            val_unit_to_g_ineq_unit=val_unit_to_g_ineq_unit,
            aux_h_eq_func=None, 
            val_unit_to_h_eq_unit=None,
            aux_dofs_init={aux_argname: lambda qp, dpfs: jnp.abs(func(qp, dofs))},
            compatibility=['f', '<='], 
            desc_unit=desc_unit,
        )