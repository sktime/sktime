"""Debugging tools."""


def eprint(*args, **kwargs):
    """Force flush."""
    print(*args, flush=True, **kwargs)


def dump_obj(msg, objname, obj):
    """Dump info with buffer flush."""
    eprint(f"\n=== {msg} ===")
    eprint(f"obj name: {objname}")
    eprint(f"type({objname}): {type(obj)}")
    if obj is None:
        return
    try:
        import pandas as pd

        if isinstance(obj, (pd.Series, pd.DataFrame)):
            eprint(f"shape({objname}): {getattr(obj, 'shape', None)}")
            eprint(f"index type: {type(obj.index)}")
            eprint(f"names: {getattr(obj.index, 'names', None)}")
            if getattr(obj, "index", None) is not None and hasattr(obj.index, "levels"):
                try:
                    eprint(f"index levels: {[list(l)[:5] for l in obj.index.levels]}")
                except Exception:
                    raise ValueError("dump_utils.dump_obj failed to dump trace")
            if isinstance(obj, pd.DataFrame):
                eprint(f"columns: {list(obj.columns)}")
                try:
                    eprint(f"dtypes: {obj.dtypes.to_dict()}")
                except Exception:
                    raise ValueError("dump_utils.dump_obj failed to dump trace")
            eprint("\n", obj)
        else:
            # numpy or other
            import numpy as np

            arr = np.asarray(obj)
            eprint(f"asarray shape: {arr.shape}  dtype: {arr.dtype}")
            eprint(f"{objname}: {arr}")
    except Exception as ex:
        eprint(f"(dump error: {ex})")
