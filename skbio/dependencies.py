# A lazy importing strategy heavily inspired by polars.
# https://github.com/pola-rs/polars/

from importlib import import_module
from importlib.util import find_spec
import re
import sys
from types import ModuleType
from typing import ClassVar, Any

_NUMPY_AVAILABLE = True
_PANDAS_AVAILABLE = True


class _LazyModule(ModuleType):
    """Module that can act both as a lazy-loader and as a proxy.

    Notes
    -----
    We do NOT register this module with ``sys.modules`` so as not to cause
    confusion in the global environment. This way we have a valid proxy
    module for our own use, but it lives *exclusively* within scikit-bio.
    """

    __lazy__ = True

    _mod_pfx: ClassVar[dict[str, str]] = {
        "numpy": "np.",
        "pandas": "pd.",
    }

    def __init__(self, module_name: str, *, module_available: bool) -> None:
        """
        Initialize lazy-loading proxy module.

        Parameters
        ----------
        module_name : str
            The name of the module to lazy-load (if available).
        module_available : bool
            Indicate if the referenced module is actually available (we will proxy it
            in both cases, but raise a helpful error when invoked if it doesn't exist).
        """
        self._module_available = module_available
        self._module_name = module_name
        self._globals = globals()
        super().__init__(module_name)

    def _import(self) -> ModuleType:
        # Import the referenced module, replacing the proxy in this module's globals.
        module = import_module(self.__name__)
        self._globals[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, name: str) -> Any:
        # Have "hasattr('__wrapped__')" return False without triggering import.
        # It's for decorators, not modules, but keeps "make doctest" happy)
        if name == "__wrapped__":
            msg = f"{self._module_name!r} object has no attribute {name!r}"
            raise AttributeError(msg)

        # Accessing the proxy module's attributes triggers import of the real thing.
        if self._module_available:
            # Import the module and return the requested attribute.
            module = self._import()
            return getattr(module, name)

        # User has not installed the proxied/lazy module.
        elif name == "__name__":
            return self._module_name
        elif re.match(r"^__\w+__$", name) and name != "__version__":
            # Allow some minimal introspection on private module
            # attrs to avoid unnecessary error-handling elsewhere.
            return None
        else:
            # All other attribute access raises a helpful exception.
            pfx = self._mod_pfx.get(self._module_name, "")
            msg = f"{pfx}{name} requires {self._module_name!r} module to be installed."
            raise ModuleNotFoundError(msg) from None


def _lazy_import(module_name: str) -> tuple[ModuleType, bool]:
    """
    Lazy import the given module; avoids up-front import costs.

    Parameters
    ----------
    module_name : str
        Name of the module to import, eg: "pandas".

    Notes
    -----
    If the requested module is not available, a proxy module
    is created in its place, which raises an exception on any
    attribute access. This allows for import and use as normal, without
    requiring explicit guard conditions - if the module is never used,
    no exception occurs; if it is, then a helpful exception is raised.

    Returns
    -------
    tuple of (Module, bool)
        A lazy-loading module and a boolean indicating if the requested/underlying
        module exists (if not, the returned module is a proxy).
    """
    # Check if module is LOADED.
    if module_name in sys.modules:
        return sys.modules[module_name], True

    # Check if module is AVAILABLE.
    try:
        module_spec = find_spec(module_name)
        module_available = not (module_spec is None or module_spec.loader is None)
    except ModuleNotFoundError:
        module_available = False

    # Create lazy/proxy module that imports the real one on first use
    # or raises an explanatory ModuleNotFoundError if not available.
    return (
        _LazyModule(
            module_name=module_name,
            module_available=module_available,
        ),
        module_available,
    )


numpy, _NUMPY_AVAILABLE = _lazy_import("numpy")
pandas, _PANDAS_AVAILABLE = _lazy_import("pandas")

__all__ = [
    "pandas",
    "numpy",
]
