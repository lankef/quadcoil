import pkgutil
import sys

from .math_utils import *
from .quadcoil_params import *
from .surface import *
from .winding_surface import *
from .wrapper import *
from .nescoil import *
from .quadcoil import *
from . import io
from . import quantities
from . import solvers
# from .conf import *
# All submodules uses quadcoil. So, we will
# not import them here to avoid circular 
# imports.

# Backward-compatible aliases for the old package names.
quantity = quantities
solver = solvers
sys.modules[__name__ + '.quantity'] = quantities
sys.modules[__name__ + '.solver'] = solvers


def _alias_submodules(canonical, legacy_name):
    """Register legacy submodule paths as the same objects as the new package."""
    for mod in pkgutil.iter_modules(canonical.__path__, canonical.__name__ + '.'):
        short = mod.name.rsplit('.', 1)[-1]
        legacy_fullname = f'{__name__}.{legacy_name}.{short}'
        # Import so both names share one module object.
        imported = __import__(mod.name, fromlist=['_'])
        sys.modules[legacy_fullname] = imported
        setattr(sys.modules[__name__ + '.' + legacy_name], short, imported)


_alias_submodules(quantities, 'quantity')
_alias_submodules(solvers, 'solver')
