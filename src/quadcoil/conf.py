# A singleton that stores some configurations.
import contextlib

class _Config:
    '''
    Global configuration manager for Quadcoil.

    The ``_Config`` class provides a global, mutable configuration interface
    for controlling Quadcoil behavior, such as the computational backend
    (e.g., ``'numpy'``, ``'jax'``) or numerical precision settings. The
    configuration can be modified globally or temporarily within a context
    using the :meth:`use` context manager.

    Parameters
    ----------
    None
        This class is not intended to be instantiated directly by users.
        A global instance is available as :data:`quadcoil.config.config`.

    Attributes
    ----------
    smoothing : str
        The smoothing method to use. The current default is 'slack', 
        which converts C0 piecewise functions to C1 by adding slack 
        variables. The alternativs are:
        - ``'approx'``: Uses the smooth approximations for these functions.
        - ``None``: Do not perform any smoothing.
    log_sum_exp_param : float
        The parameter for the LSE approximation of L-inf norms.
    huber_param : float
        The parameter for the Huber approximation of L1 norms.

    Methods
    -------
    set(key, value)
        Set a configuration parameter.
    get(key)
        Get the value of a configuration parameter.
    use(**kwargs)
        Context manager for temporarily overriding configuration values.

    Examples
    --------
    Globally set the smoothing method:

    >>> from quadcoil import config
    >>> config.set('smoothing', 'approx')
    >>> config.get('smoothing')
    'approx'

    Temporarily override the backend within a context:

    >>> from quadcoil import config
    >>> config.set('smoothing', 'slack')
    >>> with config.use(smoothing='approx'):
    ...     print(config.get('backend'))
    approx
    >>> print(config.get('backend'))
    slack

    '''
    def __init__(self):
        # Default values here
        self.smoothing = 'slack' 
        self.log_sum_exp_param = 1e-3
        self.huber_param = 1e-3

    def set(self, key, value):
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def get(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    @contextlib.contextmanager
    def use(self, **kwargs):
        
        old = {k: getattr(self, k) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setattr(self, k, v)
            yield
        finally:
            for k, v in old.items():
                setattr(self, k, v)

config = _Config()