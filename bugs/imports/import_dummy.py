import subprocess, sys, textwrap

  shapes = [
      "import sktime.transformations.series.detrend.mstl as m; assert m.MSTL.__name__ == 'MSTL'",
      "from sktime.transformations.series.detrend.mstl import MSTL; assert MSTL.__name__ == 'MSTL'",
      "import sktime.transformations.series.detrend.mstl as _; "
      "import sktime; assert sktime.transformations.series.detrend.mstl.MSTL.__name__ == 'MSTL'",
      "from sktime.transformations.series.detrend import mstl; assert mstl.MSTL.__name__ == 'MSTL'",
      "from sktime.transformations.series.detrend import MSTL; assert MSTL.__name__ == 'MSTL'",
      "from sktime.transformations.series import detrend; assert detrend.mstl.MSTL.__name__ == 'MSTL'",
      "from sktime.transformations.series.detrend.mstl import *; assert MSTL.__name__ == 'MSTL'",
      "import importlib; "
          "assert importlib.import_module('sktime.transformations.series.detrend.mstl').MSTL.__name__ == 'MSTL'",
      "import inspect, sktime.transformations.series.detrend.mstl as m; assert inspect.ismodule(m)",
  ]

  for s in shapes:
      r = subprocess.run([sys.executable, "-c", s], capture_output=True, text=True)
      print(("OK  " if r.returncode == 0 else "FAIL"), s)
      if r.returncode: print(r.stderr)
