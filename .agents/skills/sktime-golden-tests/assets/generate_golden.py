"""Generate golden reference outputs from the original <Model> implementation.

Template for the sktime-golden-tests skill. Fill in the TODOs.

Requirements for the finished script:

* it calls upstream's own code directly, and never imports sktime,
* it mirrors the sktime adapter's input construction, parameters and point reduction,
* upstream is pinned, by commit or by ``revision=``.

Source: <repo URL> @ <commit> (or HF revision)
"""

import platform
import random
from importlib.metadata import PackageNotFoundError, version

import numpy as np

# TODO: import upstream's classes here, e.g.
# from upstream_package import UpstreamModel

SEED = 42

# TODO: list the packages that determine the output
PACKAGES = ["numpy", "torch", "transformers"]

# Values of sktime.datasets.load_airline, monthly from 1949-01, embedded so this
# script has no dependency on sktime.
AIRLINE_VALUES = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]  # fmt: skip


def set_seed(seed):
    """Seed every source of randomness, where the adapter does.

    Usually right before the model call. If weights are randomly initialised at
    construction, call it before building the model as well.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def print_environment():
    """Print the versions the reference values depend on."""
    print("python", platform.python_version())
    for package in PACKAGES:
        try:
            print(package, version(package))
        except PackageNotFoundError:
            print(package, "not installed")


def run_case(context_length=None, prediction_length=12, holdout=12):
    """Return a zero-shot forecast of the airline series from upstream's code.

    Forecast the full horizon the sktime test uses, so autoregressive or multi-pass
    paths run as they do in the adapter. The test then stores only a small slice.
    """
    y = np.asarray(AIRLINE_VALUES, dtype=np.float32)
    if holdout:
        y = y[:-holdout]
    if context_length is not None:
        y = y[-context_length:]

    # TODO: load upstream's model on CPU, pinned, in eval mode, e.g.
    # model = UpstreamModel.from_pretrained("<org>/<repo>", revision="<commit>")
    # model.eval()

    set_seed(SEED)

    # TODO: run inference exactly as the adapter does, e.g.
    # with torch.no_grad():
    #     output = model.predict(y, horizon=prediction_length)
    # return output.detach().float().cpu().numpy().flatten()
    raise NotImplementedError("fill in the upstream call")


if __name__ == "__main__":
    print_environment()

    # TODO: one entry per test case
    cases = [
        {"context_length": None},
        {"context_length": 48},
    ]

    for case in cases:
        first = np.asarray(run_case(**case), dtype=np.float64)
        second = np.asarray(run_case(**case), dtype=np.float64)
        if not np.array_equal(first, second):
            raise RuntimeError(f"Not reproducible for {case}: {first} != {second}")
        # head, plus the tail when later steps take a different path
        print(case, "head", [float(value) for value in first[:3]])
        print(case, "tail", [float(value) for value in first[-3:]])
