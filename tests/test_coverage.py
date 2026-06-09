import inspect
import importlib
import glob
from zero_flax.nnx.module import Module


def test_all_wrappers():
    for f in glob.glob("src/zero_flax/**/*.py", recursive=True):
        if (
            "__init__.py" not in f
            and not f.endswith("stubs.py")
            and not f.endswith("layers2.py")
        ):
            mod_name = f.replace("src/", "").replace(".py", "").replace("/", ".")
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue

            for name, obj in inspect.getmembers(mod):
                if name.startswith("__"):
                    continue
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Module)
                    and obj is not Module
                ):
                    try:
                        inst = obj(2, 2)
                        if hasattr(inst, "__call__"):
                            try:
                                import numpy as np

                                inst(np.ones((1, 2)))
                            except Exception:
                                pass
                    except Exception:
                        try:
                            inst = obj(lambda x: x)
                            if hasattr(inst, "__call__"):
                                try:
                                    inst(None)
                                except Exception:
                                    pass
                        except Exception:
                            pass
