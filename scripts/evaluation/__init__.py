import pkgutil
import importlib

__path__ = pkgutil.extend_path(__path__, __name__)

# Automatically import all submodules
for module_info in pkgutil.walk_packages(__path__, f"{__name__}."):
    module_name = module_info.name
    importlib.import_module(module_name)
