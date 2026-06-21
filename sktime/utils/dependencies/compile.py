"""Utility to compile Cython extensions dynamically on-demand."""

import importlib
from pathlib import Path

_has_compiler_cache = None


def has_compiler():
    """Check if a working C compiler is available in the environment."""
    global _has_compiler_cache
    if _has_compiler_cache is not None:
        return _has_compiler_cache

    import os
    import shutil
    import tempfile
    from contextlib import redirect_stderr, redirect_stdout
    from pathlib import Path

    tmpdir = tempfile.mkdtemp()
    try:
        from setuptools import Distribution, Extension

        c_code = "int dummy_func(void) { return 0; }\n"
        c_file = Path(tmpdir) / "dummy.c"
        c_file.write_text(c_code)

        ext = Extension("dummy_ext", [str(c_file)])
        dist = Distribution({"ext_modules": [ext]})
        build_ext = dist.get_command_obj("build_ext")
        build_ext.inplace = 1

        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                build_ext.ensure_finalized()
                build_ext.run()
        _has_compiler_cache = True
    except Exception:
        _has_compiler_cache = False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return _has_compiler_cache


def import_or_compile_extension(module_name, pyx_relative_path, parent_file_path):
    """Import the Cython extension, or dynamically compile it in cache if missing.

    Parameters
    ----------
    module_name : str
        The name of the compiled module (e.g. '_rocket_cython_kernel').
    pyx_relative_path : str
        Relative path to the .pyx file from parent_file_path.
    parent_file_path : str
        __file__ from the calling module.
    """
    import shutil
    import sys
    import tempfile

    parent_dir = Path(parent_file_path).parent.resolve()
    pyx_path = parent_dir / pyx_relative_path

    # Try standard import first (already built or in sys.path)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    # Ensure cache directory is in sys.path
    cache_dir = Path(tempfile.gettempdir()) / "sktime_compile_cache"
    if str(cache_dir) not in sys.path:
        sys.path.append(str(cache_dir))

    # Re-try importing from the cache
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass

    if not has_compiler():
        raise ImportError(
            f"No working C compiler found. Failed to dynamically compile "
            f"or load Cython extension '{module_name}'. "
            "Please ensure a C compiler (GCC, MSVC, Clang) and 'Cython' "
            "are installed in your environment."
        )

    # Compile dynamically in the cache directory
    try:
        import numpy
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension

        # Replicate the package directory structure inside cache_dir
        module_parts = module_name.split(".")
        package_parts = module_parts[:-1]

        # Create directories and __init__.py files
        current = cache_dir
        for part in package_parts:
            current = current / part
            current.mkdir(exist_ok=True)
            init_file = current / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")

        dest_pyx_path = current / pyx_relative_path

        # Copy the .pyx source file to the user-writable cache dir
        shutil.copy2(pyx_path, dest_pyx_path)

        ext = Extension(
            module_name, [str(dest_pyx_path)], include_dirs=[numpy.get_include()]
        )

        # Build programmatically inplace
        dist = Distribution({"ext_modules": cythonize([ext], quiet=True)})
        build_ext = dist.get_command_obj("build_ext")
        build_ext.inplace = 1
        build_ext.ensure_finalized()
        build_ext.run()

        # Re-attempt import
        return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(
            f"Failed to dynamically compile or load Cython extension '{module_name}'. "
            "Please ensure a C compiler and 'Cython' are installed in your environment."
        ) from e
