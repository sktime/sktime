# vmdpy: Variational mode decomposition in Python

Function for decomposing a signal according to the Variational Mode Decomposition
([Dragomiretskiy and Zosso, 2014](https://doi.org/10.1109/TSP.2013.2288675)) method.

This package is a Python translation of
the original [VMD MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition)


## Installation

`vmdpy` is distributed with `sktime`.

Install ``sktime`` via ``pip`` or ``conda``, i.e.,

```
pip install sktime
```

or

```
conda install sktime
```

For further details, see the [sktime installation guide](https://www.sktime.net/en/stable/installation.html)


## Citation and Contact
Paper available at: https://doi.org/10.1016/j.bspc.2020.102073

If you find this package useful, we kindly ask you to cite it in your work:
Vinícius R. Carvalho, Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes,
Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification,
Biomedical Signal Processing and Control,
Volume 62,
2020,
102073,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2020.102073.

For contributing new functionality or fixing anything in the package,
kindly make a PR to the ``sktime`` repository (``libs.vmdpy`` module).

For suggestions, questions, comments:

* [sktime issue tracker](https://github.com/sktime/sktime/issues) or [discussion forum](https://github.com/sktime/sktime/discussions),
  please ping `vrcarva`
* [sktime discord](https://discord.com/invite/54ACzaFsn7)


## Example script
```python
#%% Simple example: generate signal with 3 components + noise
import numpy as np
import matplotlib.pyplot as plt
from sktime.libs.vmdpy import VMD

# Time Domain 0 to T
T = 1000
fs = 1 / T
t = np.arange(1, T + 1) / T
freqs = 2 * np.pi * (t - 0.5 - fs) / (fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = np.cos(2 * np.pi * f_1 * t)
v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))

f = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)

# some sample parameters for VMD
alpha = 2000  # moderate bandwidth constraint
tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
K = 3  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

# Run VMD
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

# Visualize decomposed modes
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(f)
plt.title("Original signal")
plt.xlabel("time (s)")
plt.subplot(2, 1, 2)
plt.plot(u.T)
plt.title("Decomposed modes")
plt.xlabel("time (s)")
plt.legend(["Mode %d" % m_i for m_i in range(u.shape[0])])
plt.tight_layout()

```
