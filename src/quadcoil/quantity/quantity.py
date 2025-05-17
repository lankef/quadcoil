from typing import Callable, List, Any
from jax import jit
import jax.numpy as jnp
from functools import partial

def _take_stellsym(field):
    # First, we flatten the phi/theta dependence
    # of field. These usually are its first two axes
    shape = field.shape
    new_shape = (shape[0] * shape[1],) + shape[2:]
    flat_field = jnp.reshape(field, new_shape)
    # Then we take only roughly half of the sample points.
    # Note to developers: It's wrong to take half of the 
    # sample points naively, because our quadrature points
    # do not sample from 0 to 1!
    # Note that the simplest, but also safest choice
    # is to take (nphi//2 + 1) * ntheta elements. 
    # This will still introduce O(ntheta) duplicate
    # elements but it saves us of treating even/odd 
    # nphi/ntheta cases separately. 
    return flat_field[:(shape[0]//2+1) * shape[1]]
    
class _Quantity: 
    r'''
    Interior point methods require :math:`C^1` objectives. Therefore, 
    QUADCOIL has to convert non-:math:`C^1`, convex objectives, such as 
    L-:math:`\infty` or L1 norms, into a combination of :math:`C^1` objectives, constraints
    and auxiliary variables. For example, the L-:math:`\infty` norm 
    :math:`\max_{ws}\Phi`, is actually:
    
    .. math::
   
        \max_{ws}\Phi = \min s, \text{ where } |\Phi|\leq s.

    This "under-the-hood" implementation is less intuitive, and will not work for the optimum
    of a different problem that does not contain the same auxiliary variable. To make 
    QUADCOIL (hopefully) easier to use and develop for, we've came up with the ``_Quantity`` class.

    ``_Quantity`` is designed as a class for storing all these different forms of the same
    physical quantity in the same place. You can think of it as a ``FunctionType`` that 
    switches between: 
        * A "surface-level", non-:math:`C^1` implementation that's 
        easy to understand and works for all problems. This is ``c0_impl``.
        * An "under-the-hood" implementation :math:`C^1`, but nay not be well-defined in
        another problem with different auxiliary vars, and is less intuitive to the user.
        These are ``scaled_c2_impl, scaled_g_ineq_impl, scaled_h_eq_impl, scaled_aux_dofs_init``.

    To add a new quantity to QUADCOIL, a developer only need to create a new **instance** (not subclass!)
    of ``_Quantity`` and provide all the functions and limitations necessary. Users will import and use
    these instances like functions, and why'll work across all cases, and developers can use the internal checks in 
    ``_Quantity`` to help make sure that they've implemented things properly.

    To developers familiar with DESC, this kind of design may seem somewhat similar to
    ``_Objective``, but again, please think of instances of ``_Quantity`` as 
    functions rather than objects. It stores nether the problem setup, nor the state during a solve.
    It's purely for use by ``quadcoil/wrapper.py`` to construct the :math:`f, g, h` of the constrained 
    optimization problem.

    Quantities with auxiliary variables are often incompatible with ``'=='`` and ``'>='`` 
    constraints. For example, :math:`\max_{ws}\Phi=\alpha` and :math:`\max_{ws}\Phi\geq\alpha`
    are both non-convex constraints, and not supported by QUADCOIL. The compatibility is 
    stored in ``Objective``. 
    
    Parameters
    ----------
    scaled_c2_impl : Callable(qp: QuadcoilParams, dofs: dict)
        The value of the quantity. Must be :math:`C^1`.
    c0_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
        A "real" implementation of the quantity that can directly be calculated from :math:`\Phi_{sv}`
        without using auxiliary variables. Can be :math:`C^0` rather than :math:`C^1`. This is the 
        "surface-level" imnplementation that works across different problems, and that 
        QUADCOIL users will typically access. Scale dependence must be included.
    scaled_g_ineq_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
        The auxiliary inequality constraints required by the objective, in the form 
        of :math:`g(x)\leq0`. Note the inclusion of ``unit`` as an argument. 
        This is because the auxiliary constraint of the quantity often have a different unit from the 
        value of the quantity itself. For example, the L-1 norm of a field :math:`v` has the unit of 
        :math:`(\text{unit}_v) m^2`, whereas its auxiliary constraints have the unit of
        :math:`(\text{unit}_v)`. To allow the user to appoint one unit for the entire quantity, 
        we must provide a unit conversion function. The function will access the problem setup (but not state)
        via a ``QuadcoilParams``. **It is our responsibility as developers to make g and h unit-free!**
    scaled_h_eq_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
        The auxiliary equality constraints required by the objective, in the form 
        of :math:`h(x)=0`.  
    scaled_aux_dofs_init : dict{str: Callable(qp: QuadcoilParams, dofs: dict, unit: float) -> Tuple}
    compatibility : List[str]
        Whether this quantity can serve as an objective, or appear in
        ``'<='``  ``'=='`` and (or) ``'>='`` constraints. Must be a ``List`` containing one or 
        more of ``'f'``, ``'<='``  ``'=='`` and (or) ``'>='``.
    desc_unit : Callable(scale: dict)
        For calculating the unit of this quantity in DESC.

    Attributes
    ----------
    scaled_c2_impl : Callable(qp: QuadcoilParams, dofs: dict)
    c0_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
    scaled_g_ineq_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
    scaled_h_eq_impl : Callable(qp: QuadcoilParams, dofs: dict, unit: float)
    scaled_aux_dofs_init : dict{str: Callable(qp: QuadcoilParams, dofs: dict, unit: float) -> Tuple}
    desc_unit : Callable(scale: dict)
    compatibility : List[str]
    '''

    _used_aux_names = set()  # class-level registry for checking duplicate aux var names

    def __init__(
        self, 
        scaled_c2_impl, 
        c0_impl, 
        scaled_g_ineq_impl,
        scaled_h_eq_impl,
        scaled_aux_dofs_init, 
        compatibility, 
        desc_unit
    ):
        if isinstance(scaled_c2_impl, _Quantity):
            raise TyperError('scaled_c2_impl must not be a _Quantity.')
        if isinstance(c0_impl, _Quantity):
            raise TyperError('c0_impl must not be a _Quantity.')
        if isinstance(scaled_g_ineq_impl, _Quantity):
            raise TyperError('scaled_g_ineq_impl must not be a _Quantity.')
        if isinstance(scaled_h_eq_impl, _Quantity):
            raise TyperError('scaled_h_eq_impl must not be a _Quantity.')
        # Checking whether the auxiliary variable name has already been used elsewhere.
        if scaled_aux_dofs_init is not None:
            if not isinstance(scaled_aux_dofs_init, dict):
                raise TypeError('scaled_aux_dofs_init muse be a dict.')
            aux_names = set(scaled_aux_dofs_init.keys())
            dup_aux_names = self.__class__._used_aux_names.intersection(aux_names)
            # Check for duplicates
            if dup_aux_names:
                raise ValueError(
                    f'The auxiliary variable name {dup_aux_names} '\
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
        self.scaled_c2_impl = scaled_c2_impl
        self.c0_impl = c0_impl
        self.scaled_g_ineq_impl = scaled_g_ineq_impl
        self.scaled_h_eq_impl = scaled_h_eq_impl
        self.scaled_aux_dofs_init = scaled_aux_dofs_init
        self.compatibility = compatibility
        self.desc_unit = desc_unit
        
    @partial(jit, static_argnames=('self',))
    def __call__(self, qp, dofs):
        r'''
        Evaluate this quantity. For  convenience.
        
        Parameters
        ----------
        qp : QuadcoilParams
            (static) The problem setup, including the plasma surface, 
            the winding surface, the currents, but not the objective/constraint choices.
        dofs : dict
            (static) A dictionary storing the degrees of freedom in a QUADCOIL problem. 
        '''
        return self.c0_impl(qp, {'phi': dofs['phi']})

    def generate_c2(func, compatibility, desc_unit):
        return _Quantity(
            scaled_c2_impl=lambda qp, dofs, unit: func(qp, dofs)/unit, 
            c0_impl=func, 
            scaled_g_ineq_impl=None,
            scaled_h_eq_impl=None,
            scaled_aux_dofs_init=None, 
            compatibility=compatibility, 
            desc_unit=desc_unit,
        )

    def generate_linf_norm(func, aux_argname, desc_unit, positive_definite=False, square=False, auto_stellsym=False):
        r'''
        Generates the auxiliary constraints for an L-:math:`\infty`
        norm. See documentations for ``quadcoil.objective.Objective``.
        An L-:math:`\infty` norm term in an objective or :math:`\leq`
        constraint can be represented by:

        .. math::

            max|f| &= p (\text{or } p^2 \text{ if square=True}),\\
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
            The name of the auxiliary variable. Must be unique among all supported
            ``_Quantity``s.
        desc_unit : Callable
            A callable calculating the quantity's unit in DESC.
        positive_definite : bool, optional, default=False
            Whether ``func`` is positive definite. If ``True``, then
            the second constraint is not required.
        square : bool, optional, default=True
            When ``True``, generates :math:`\|f\|_\infty^2` instead. This is better-behaved
            when used as objective.
        auto_stellsym : bool, optional, default=False
            When ``True``, ignores the second half of all objective values when constructing 
            constraints. Reduces computational cost and improves conditioning.

        Returns
        -------
        A ``_Quantity``.
        '''
        # The objective/constraint form of this L-infinity
        # norm is just a function that returns the auxiliary variable.
        # Because the aux var is already scaled to O(1),
        # scaled_c2_impl doesn't acually need to have unit dependence.
        if square:
            scaled_c2_impl = lambda qp, dofs, unit, aux_argname=aux_argname: dofs[aux_argname]**2
            # The effective function
            c0_impl = lambda qp, dofs, func=func: jnp.max(jnp.abs(func(qp, dofs)))**2
        else:
            scaled_c2_impl = lambda qp, dofs, unit, aux_argname=aux_argname: dofs[aux_argname]
            # The effective function
            c0_impl = lambda qp, dofs, func=func: jnp.max(jnp.abs(func(qp, dofs)))
        # The constraints
        def scaled_g_ineq_impl(qp, dofs, unit, func=func, aux_argname=aux_argname):
            # When square==True, the specified by the user is field^2.
            if square:
                unit_eff = jnp.sqrt(unit)
            else:
                unit_eff = unit
            field = func(qp, dofs)
            if auto_stellsym and qp.stellsym:
                field = _take_stellsym(field)
            p_aux = dofs[aux_argname]
            g_plus = field/unit_eff - p_aux #  f - p <=0
            # We need only half of the constraints if f is positive definite
            if positive_definite:
                return g_plus
            g_minus = -field/unit_eff - p_aux # -f - p <=0
            return jnp.stack([g_plus,g_minus], axis=0)
        
        # The initial value of the auxiliary variable is the 
        # c0_impl / unit. 
        if square:
            scaled_aux_dofs_init = lambda qp, dofs, unit: c0_impl(qp, dofs)/jnp.sqrt(unit)
        else:
            scaled_aux_dofs_init = lambda qp, dofs, unit: c0_impl(qp, dofs)/unit
        return _Quantity(
            scaled_c2_impl=scaled_c2_impl,
            c0_impl=c0_impl, 
            scaled_g_ineq_impl=scaled_g_ineq_impl,
            scaled_h_eq_impl=None, 
            # The auxiliary dofs' names and initial values
            scaled_aux_dofs_init={aux_argname: scaled_aux_dofs_init},
            compatibility=['f', '<='], 
            desc_unit=desc_unit,
        )
    
    def generate_linf_norm_4(func, aux_argname, desc_unit):
        r'''
        Generates the auxiliary constraints for an L-:math:`\infty`
        norm. See documentations for ``quadcoil.objective.Objective``.
        An L-:math:`\infty` norm term in an objective or :math:`\leq`
        constraint can be represented by:

        .. math::

            max|f|^4 &= p^2,\\
            \text{where}&\\
             f^2 - p &\leq 0\\
        
        This is a scalar, convex quantity that's only compatible as an 
        objective term or with ``'<='`` constraints.

        Parameters
        ----------  
        func : Callable
            The source function :math:`f` to convert into an L-:math:`\infty` norm.
        aux_argname : str
            The name of the auxiliary variable. Must be unique among all supported
            ``_Quantity``s.
        desc_unit : Callable
            A callable calculating the quantity's unit in DESC.
        positive_definite : bool, optional, default=False
            Whether ``func`` is positive definite. If ``True``, then
            the second constraint is not required.
        square : bool, optional, default=True
            When ``True``, generates :math:`\|f\|_\infty^2` instead. This is better-behaved
            when used as objective.

        Returns
        -------
        A ``_Quantity``.
        '''
        # The objective/constraint form of this L-infinity
        # norm is just a function that returns the auxiliary variable.
        # Because the aux var is already scaled to O(1),
        # scaled_c2_impl doesn't acually need to have unit dependence.
        scaled_c2_impl = lambda qp, dofs, unit, aux_argname=aux_argname: dofs[aux_argname]**2
        # The effective function
        c0_impl = lambda qp, dofs, func=func: jnp.max(func(qp, dofs)**4)
        # The constraints
        def scaled_g_ineq_impl(qp, dofs, unit, func=func, aux_argname=aux_argname):
            # The unit specified by the user is field^4
            unit_sq = jnp.sqrt(unit)
            field = func(qp, dofs)
            p_aux = dofs[aux_argname]
            g_sq = field**2/unit_sq - p_aux
            return g_sq
        
        # The initial value of the auxiliary variable is the 
        # max(|field|^2) / unit_sq. 
        scaled_aux_dofs_init = lambda qp, dofs, unit: jnp.max(func(qp, dofs)**2)/jnp.sqrt(unit)
        return _Quantity(
            scaled_c2_impl=scaled_c2_impl,
            c0_impl=c0_impl, 
            scaled_g_ineq_impl=scaled_g_ineq_impl,
            scaled_h_eq_impl=None, 
            # The auxiliary dofs' names and initial values
            scaled_aux_dofs_init={aux_argname: scaled_aux_dofs_init},
            compatibility=['f', '<='], 
            desc_unit=desc_unit,
        )
    
    def generate_l1_norm(func, aux_argname, desc_unit, positive_definite=False, square=False, auto_stellsym=False):
        r'''
        Generates the auxiliary constraints for an L-1 
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
            The name of the auxiliary variable. Must be unique among all supported
            ``_Quantity``s.
        maxdesc_unit : Callable
            A callable calculating the quantity's unit in DESC.
        positive_definite : bool, optional, default=False
            Whether ``func`` is positive definite. If ``True``, then
            the second constraint is not required.
        square : bool, optional, default=True
            When ``True``, generates :math:`\|f\|_1^2` instead. For non-singular Hessians.
        auto_stellsym : bool, optional, default=False
            When ``True``, ignores the second half of all objective values when constructing 
            constraints. Reduces computational cost and improves conditioning.

        Returns
        -------
        A ``_Quantity``.
        '''
        # The objective/constraint form of this L-1
        # norm is the surface integral pf the aux variable.
        def scaled_c2_impl(qp, dofs, unit, aux_argname=aux_argname):
            # Because the aux vars are already
            # scaled to O(1), no unit dependence is actually needed here.
            da_flat = qp.eval_surface.da()
            if auto_stellsym and qp.stellsym:
                da_flat = _take_stellsym(da_flat)
            return jnp.sum(da_flat * dofs[aux_argname]) * qp.nfp
        # The effective function
        def c0_impl(qp, dofs, func=func):
            raise NotImplementedError('Not working yet, please don\'t use')
            da_flat = qp.eval_surface.da()
            field = func(qp, dofs)
            if auto_stellsym and qp.stellsym:
                da_flat = _take_stellsym(da_flat)
                field = _take_stellsym(field)
            return jnp.sum(da_flat * field) * qp.nfp
        # The constraints
        def scaled_g_ineq_impl(qp, dofs, unit, func=func, aux_argname=aux_argname, auto_stellsym=auto_stellsym):
            field = func(qp, dofs)
            if auto_stellsym and qp.stellsym:
                field = _take_stellsym(field)
            p_aux = dofs[aux_argname]
            g_plus = field/unit - p_aux #  f - p <=0
            # We need only half of the constraints if f is positive definite
            if positive_definite:
                return g_plus
            g_minus = -field/unit - p_aux # -f - p <=0
            return jnp.stack([g_plus,g_minus], axis=0)
        # The unit of L-1 norm is m^2 (unit),
        # but the unit of its auxiliary constraints
        # are (unit). Therefore, g_unit is val_unit \
        # divided by the surface's area.
        def g_unit(qp, unit):
            da_flat = qp.eval_surface.da()
            if auto_stellsym and qp.stellsym:
                da_flat = _take_stellsym(da_flat)
            return unit / (jnp.sum(da_flat) * qp.nfp)
        # The initial value of the auxiliary variable is the 
        # abs(field) / g_unit. 
        def scaled_aux_dofs_init(qp, dofs, unit, auto_stellsym=auto_stellsym):
            field = func(qp, dofs)
            if auto_stellsym and qp.stellsym:
                field = _take_stellsym(field)
            return jnp.abs(field)/g_unit(qp, unit)
        return _Quantity(
            scaled_c2_impl=scaled_c2_impl,
            c0_impl=c0_impl, 
            scaled_g_ineq_impl=scaled_g_ineq_impl,
            scaled_h_eq_impl=None, 
            scaled_aux_dofs_init={aux_argname: scaled_aux_dofs_init},
            compatibility=['f', '<='], 
            desc_unit=desc_unit,
        )