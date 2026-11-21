# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Measure HilbertTransformer runtime and traced peak allocations.

Run from an editable sktime checkout:
    python _contrib/benchmark_hilbert.py --output hilbert-benchmark.csv

The reference class below contains the original per-channel Hilbert algorithm
from sktime/transformations/hilbert.py, prior to issue #11375's optimization.
Both variants are measured through the public fit_transform API.
"""

import argparse
import csv
import gc
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert

from sktime.transformations.hilbert import HilbertTransformer


class PerChannelHilbert(HilbertTransformer):
    """Use the original per-channel algorithm as the end-to-end reference."""

    def _transform(self, X, y=None):
        self._check_params()
        was_series = isinstance(X, pd.Series)
        df = pd.DataFrame(X) if was_series else X
        n = len(df)
        if self.N is not None and self.N < n:
            raise ValueError(
                f"N ({self.N}) must be greater than or equal to signal length ({n})."
            )
        result = {}
        for col in df.columns:
            arr = df[col].to_numpy()
            z = hilbert(arr, N=self.N, axis=0)[:n]
            envelope = np.abs(z)
            phase = np.angle(z)
            if self.unwrap_phase:
                phase = np.unwrap(phase)
            if self.output_type == "envelope":
                result[col] = envelope
            elif self.output_type == "quadrature":
                result[col] = np.imag(z)
            elif self.output_type == "phase":
                result[col] = phase
            elif self.output_type == "frequency":
                result[col] = np.gradient(phase) * self.fs / (2 * np.pi)
            elif self.output_type == "all":
                freq = np.gradient(phase) * self.fs / (2 * np.pi)
                result[f"{col}_envelope"] = envelope
                result[f"{col}_phase"] = phase
                result[f"{col}_frequency"] = freq
        out = pd.DataFrame(result, index=df.index)
        if was_series and self.output_type != "all":
            return pd.Series(out.iloc[:, 0], index=X.index, name=X.name)
        return out


def transform(cls, X, mode):
    """Measure public fit_transform, including input/output handling."""
    return cls(output_type=mode, fs=4.0).fit_transform(X)


def median_runtime_ms(a, b, X, mode, repeats):
    """Time variants with warm-ups and alternating execution order."""
    timings = {"old": [], "new": []}
    pair = (("old", a), ("new", b))
    for cls in (a, b):
        for _ in range(2):
            transform(cls, X, mode)
    for rep in range(repeats):
        for label, cls in pair if rep % 2 == 0 else reversed(pair):
            gc.collect()
            start = time.perf_counter()
            transform(cls, X, mode)
            timings[label].append(1000 * (time.perf_counter() - start))
    return statistics.median(timings["old"]), statistics.median(timings["new"])


def peak_traced_mib(cls, X, mode, repeats):
    """Median peak tracemalloc allocation; not a measurement of process RSS."""
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        try:
            transform(cls, X, mode)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peaks.append(peak / (1024 * 1024))
    return statistics.median(peaks)


def main():
    """Run benchmarks on representative channel counts and save optional CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--channels", type=int, nargs="+", default=[1, 8, 32, 96, 256])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.length < 4 or args.repeats < 2 or args.memory_repeats < 1:
        parser.error("length >= 4, repeats >= 2, memory-repeats >= 1 required")
    if not args.channels or min(args.channels) < 1:
        parser.error("at least one positive channel count required")

    print(
        f"Python {platform.python_version()}; NumPy {np.__version__}; "
        f"pandas {pd.__version__}; SciPy {scipy.__version__}"
    )
    print(
        f"Timepoints={args.length}; seed=2026; "
        f"timing repeats={args.repeats}; memory repeats={args.memory_repeats}"
    )
    print("Peak memory is tracemalloc-tracked allocation, not process RSS.\n")

    rng = np.random.default_rng(2026)
    data = rng.standard_normal((args.length, max(args.channels)))
    rows = []
    print(
        f"{'Mode':<12}{'Channels':>9}{'Old ms':>12}{'New ms':>12}"
        f"{'Speedup':>10}{'Old MiB':>11}{'New MiB':>11}"
    )
    print("-" * 77)
    for mode in ("envelope", "all", "frequency"):
        for channels in args.channels:
            X = pd.DataFrame(
                data[:, :channels].copy(),
                columns=[f"c_{i}" for i in range(channels)],
            )
            old = transform(PerChannelHilbert, X, mode)
            new = transform(HilbertTransformer, X, mode)
            pd.testing.assert_frame_equal(old, new, rtol=1e-12, atol=1e-12)
            old_ms, new_ms = median_runtime_ms(
                PerChannelHilbert, HilbertTransformer, X, mode, args.repeats
            )
            old_mib = peak_traced_mib(PerChannelHilbert, X, mode, args.memory_repeats)
            new_mib = peak_traced_mib(HilbertTransformer, X, mode, args.memory_repeats)
            rows.append(
                {
                    "mode": mode,
                    "timepoints": args.length,
                    "channels": channels,
                    "baseline_ms": round(old_ms, 5),
                    "optimized_ms": round(new_ms, 5),
                    "speedup": round(old_ms / new_ms, 4),
                    "baseline_traced_mib": round(old_mib, 5),
                    "optimized_traced_mib": round(new_mib, 5),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scipy": scipy.__version__,
                }
            )
            print(
                f"{mode:<12}{channels:>9}{old_ms:>12.3f}{new_ms:>12.3f}"
                f"{old_ms / new_ms:>9.2f}x{old_mib:>11.3f}{new_mib:>11.3f}"
            )
    if args.output:
        path = args.output.expanduser()
        with path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {path}")
    print("All baseline/optimized numerical comparisons passed.")


if __name__ == "__main__":
    main()
