import importlib

_lazy: dict[str, tuple[str, str | None]] = {
    'World': ('stable_worldmodel.world', 'World'),
    'PlanConfig': ('stable_worldmodel.policy', 'PlanConfig'),
    'pretraining': ('stable_worldmodel.utils', 'pretraining'),
    'data': ('stable_worldmodel.data', None),
    'envs': ('stable_worldmodel.envs', None),
    'policy': ('stable_worldmodel.policy', None),
    'solver': ('stable_worldmodel.solver', None),
    'spaces': ('stable_worldmodel.spaces', None),
    'utils': ('stable_worldmodel.utils', None),
    'wm': ('stable_worldmodel.wm', None),
    'wrapper': ('stable_worldmodel.wrapper', None),
}

__all__ = list(_lazy)


def __getattr__(name: str):
    if name in _lazy:
        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr) if attr else mod
        globals()[name] = val
        return val
    raise AttributeError(f"module 'stable_worldmodel' has no attribute {name!r}")
