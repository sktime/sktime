"""Datasets loaders for pykalman."""

from os.path import dirname, join

import numpy as np
from numpy import ma
from scipy import io

from ..utils import Bunch


def load_robot():
    """Load and return synthetic robot state data (state estimation).

    =================================
    Number of time steps          501
    Dimensionality of Observations  2
    Dimensionality of States        5
    =================================

    Returns
    -------
    data : Bunch
        Dictionary-like object containing all data. Access attributes as you
        would the contents of a dictionary or of an object.

    Examples
    --------
    >>> from sktime.libs.pykalman.datasets import load_robot
    >>> data = load_robot()
    >>> data.observations.shape
    (501, 2)
    """

    def pad_and_mask(X):
        """Pad X's first index with zeros and mask it."""
        zeros = np.zeros(X.shape[1:])[np.newaxis]
        X = np.vstack([zeros, X])
        mask = np.zeros(X.shape)
        mask[0] = True
        return ma.array(X, mask=mask)

    module_path = dirname(__file__)
    data = io.loadmat(join(module_path, "data", "robot.mat"))
    descr = open(join(module_path, "descr", "robot.rst")).read()
    Z = pad_and_mask(data["y"].T)
    X = data["x"].T
    A = data["A"]
    b = data["b"].T
    C = data["C"]
    d = data["d"][:, 0]
    Q_0 = 10.0 * np.eye(5)
    R_0 = 10.0 * np.eye(2)
    Q = data["Q"]
    R = data["R"]
    x_0 = data["x0"][:, 0]
    V_0 = data["P_0"]
    X_filt = data["xfilt"].T
    V_filt = data["Vfilt"][0]
    ll = data["ll"][0]
    X_smooth = data["xsmooth"].T
    V_smooth = data["Vsmooth"][0]
    T = Z.shape[0]

    # V_filt is actually an object array where each object is a 2D array.
    # Convert it to a proper, 3D array.  Likewise for V_smooth.
    V_filt = np.asarray([V_filt[t] for t in range(V_filt.shape[0])])
    V_smooth = np.asarray([V_smooth[t] for t in range(V_smooth.shape[0])])

    return Bunch(
        n_timesteps=T,
        observations=Z,
        states=X,
        transition_matrix=A,
        transition_offsets=b,
        observation_matrix=C,
        observation_offset=d,
        initial_transition_covariance=Q_0,
        initial_observation_covariance=R_0,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=x_0,
        initial_state_covariance=V_0,
        filtered_state_means=X_filt,
        filtered_state_covariances=V_filt,
        loglikelihoods=ll,
        smoothed_state_means=X_smooth,
        smoothed_state_covariances=V_smooth,
        DESCR=descr,
    )
