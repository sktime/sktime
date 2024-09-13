"""MAPA (Multiple Aggregation Prediction Algorithm) Forecaster."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.utils.datetime import _get_freq
from sktime.utils.validation.series import check_series


class MAPAForecaster(BaseForecaster):
    """MAPA (Multiple Aggregation Prediction Algorithm) Forecaster."""

    def __init__(
        self,
        comb: str = "w.mean",
        hybrid: bool = True,
        conf_lvl: Optional[float] = None,
        sp: Optional[int] = None,
        etstype: str = "ets",  # Add this line
    ):
        super().__init__()
        self.comb = comb
        self.hybrid = hybrid
        self.conf_lvl = conf_lvl
        self.sp = sp
        self.etstype = etstype  # Add this line
        self._y = None
        self._mapafit = None
        self._is_fitted = False

    @property
    def y_(self):
        """Returns _y."""
        return self._y

    @property
    def mapafit_(self):
        """Returns _mapafit."""
        return self._mapafit

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        self._y = check_series(y).copy()
        self._mapafit = self._prepare_mapafit(self._y)
        if X is not None:
            self._validate_xreg(X, len(fh) if fh is not None else 0, len(self._y))
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None):
        """Make forecasts for the given forecast horizon."""
        self._check_is_fitted()

        # Create deep copies of internal state to prevent modifications
        y_copy = self._y.copy()
        mapafit_copy = deepcopy(self._mapafit)

        result = self.mapafor(
            y_copy,
            mapafit_copy,
            fh=len(fh),
            comb=self.comb,
            hybrid=self.hybrid,
            conf_lvl=self.conf_lvl,
            xreg=X,
        )

        if len(result["outfor"]) == 0:
            return pd.Series(index=fh)  # Return an empty Series with the correct index
        else:
            return pd.Series(result["outfor"], index=fh)

    def _update(self, y, X=None, update_params=True):
        """Update the forecaster with new data."""
        if not self._is_fitted:
            return self._fit(y, X)

        self._y = pd.concat([self._y, check_series(y)])

        if update_params:
            self._mapafit = self._prepare_mapafit(self._y)

        return self

    def _prepare_mapafit(self, y):
        """Prepare MAPA fit data using AutoETS."""
        sp = self.sp if self.sp is not None else self._infer_seasonal_period(y)

        return pd.DataFrame(
            {
                "use": [True, True],
                "AL": [1, sp],
                "original.ppy": [sp, sp],
                "etstype": [self.etstype, self.etstype],
            }
        )

    def _fit_auto_ets(self, yA: pd.Series, ppyA: int) -> AutoETS:
        """Fit AutoETS model."""
        auto_ets = AutoETS(sp=ppyA, auto=True)
        return auto_ets.fit(yA)

    def _infer_seasonal_period(self, y):
        """Infer seasonal period from the data."""
        if isinstance(y.index, pd.DatetimeIndex):
            freq = y.index.freq
            if freq is not None:
                if freq.name == "M":
                    return 12
                elif freq.name == "D":
                    return 7
                elif freq.name == "H":
                    return 24
        return 1

    def _check_is_fitted(self):
        """Check if the forecaster is fitted."""
        if not self._is_fitted:
            raise ValueError(
                "Forecaster is not fitted yet. Call 'fit' before prediction."
            )

    def tsaggr(self, y, fout, fmean=True, outplot=True):
        """Perform temporal aggregation on time series data."""
        n = len(y)

        if isinstance(y, pd.Series):
            f = _get_freq(y.index)
        else:
            f = None
            y = pd.Series(y)

        if isinstance(fout, int):
            fout = [fout]
        fout = [f for f in fout if 0 < f <= n]
        if len(fout) == 0:
            raise ValueError(
                "fout must be positive integer(s), smaller than length(y)."
            )
        k = len(fout)

        scl = fout if fmean else [1] * k

        y_all = pd.DataFrame(index=y.index, columns=[f"AL{f}" for f in fout])
        y_idx = []
        y_out = []
        slost = []

        for i, f in enumerate(fout):
            slost.append(n % f)
            idx = range(slost[i], n - f + 1, f)
            y_idx.append(idx)
            nk = len(idx)

            for j in range(nk):
                start = idx[j]
                end = idx[j] + f
                y_all.iloc[start:end, i] = y.iloc[start:end].sum() / scl[i]

            y_out.append(y_all.iloc[idx, i].dropna())

            if f is not None and f % fout[i] == 0:
                y_out[i].index = pd.date_range(
                    y.index[0], periods=len(y_out[i]), freq=f"{f}T"
                )

        if outplot:
            self._plot_tsaggr(y, y_out, fout)

        return {"out": y_out, "all": y_all, "idx": y_idx}

    def _plot_tsaggr(self, y, y_out, fout):
        from sktime.utils.dependencies._dependencies import _check_soft_dependencies

        try:
            _check_soft_dependencies("matplotlib", severity="warning")
            import matplotlib.pyplot as plt
        except ImportError:
            # If matplotlib is not installed, we'll just return without plotting
            return
        plt.figure(figsize=(12, 6))
        plt.plot(y.index, y.values, color="black", linewidth=2, label="Original")
        colors = plt.cm.rainbow(np.linspace(0, 0.8, len(fout)))
        for i, (series, color) in enumerate(zip(y_out, colors)):
            plt.plot(series.index, series.values, color=color, label=f"AL{fout[i]}")
            plt.scatter(series.index, series.values, color=color, s=20)
        plt.legend(loc="upper left")
        plt.xlabel("Period")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    def statetranslate(self, fit, AL, fh, q, ppyA, fittype, xreg=None):
        """Translate the state of a fitted AutoETS model into forecast components."""
        FCs_temp = np.zeros((5, fh))
        fhA = int(np.ceil(fh / AL))

        # Extract components from the fitted AutoETS model
        level = fit._fitted_forecaster.level
        trend = fit._fitted_forecaster.trend
        season = fit._fitted_forecaster.seasonal

        # Estimates for the Level Component
        FCs_temp[1, :] = np.repeat(np.repeat(level[-1], fhA), AL)[:fh]

        # Estimates for the Trend Component
        if trend is not None:
            FCs_temp[2, :] = np.repeat(trend[-1] * np.arange(1, fhA + 1), AL)[:fh]

        # Estimates for the Seasonal Component
        if season is not None:
            FCs_temp[3, :] = np.repeat(np.tile(season[-ppyA:], fhA), AL)[:fh]

        # Estimate for the xreg (if provided)
        if xreg is not None:
            if xreg.ndim > 1:
                FCs_temp[4, :] = np.repeat(
                    np.sum(xreg * fit._fitted_forecaster.params["beta"], axis=1), AL
                )[:fh]
            else:
                FCs_temp[4, :] = np.repeat(
                    xreg * fit._fitted_forecaster.params["beta"], AL
                )[:fh]

        # Recreate forecasts if fittype is 1
        if fittype == 1:
            FCs_temp[0, :] = np.sum(FCs_temp[1:], axis=0)

        return FCs_temp

    def mapaprcomp(self, x, pr_comp):
        """Preprocess xreg with principal component analysis.

        This function applies PCA to the input data based on the specified
        number of components.
        """
        if pr_comp["pr.comp"] == 0:
            return x
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        pca = PCA(n_components=pr_comp["pr.comp"])
        x_pca = pca.fit_transform(x_scaled)
        x_out = x_pca * np.sqrt(pca.explained_variance_)
        if "mean" in pr_comp and pr_comp["mean"] is not None:
            x_out += pr_comp["mean"]
        return x_out

    def mapacalc(
        self,
        y: pd.Series,
        mapafit: pd.DataFrame,
        fh: int = 0,
        comb: str = "w.mean",
        outplot: int = 0,
        hybrid: bool = True,
        xreg: Optional[pd.DataFrame] = None,
    ) -> dict[str, np.ndarray]:
        """Calculate MAPA forecasts using AutoETS."""
        ALs = mapafit[mapafit["use"]]["AL"].values
        # minimumAL = min(ALs)
        maximumAL = max(ALs)
        ppy = mapafit["original.ppy"].iloc[0]
        # ets_type = mapafit["etstype"].iloc[0]

        if fh == 0:
            fh = ppy

        observations = len(y)

        if xreg is not None:
            self._validate_xreg(xreg, fh, observations)

        FCs = np.zeros((maximumAL - min(ALs) + 1, 5, fh))

        for ALi, AL in enumerate(range(min(ALs), maximumAL + 1)):
            fhA = int(np.ceil(fh / AL))
            q = observations // AL
            ppyA = ppy // AL if ppy % AL != 0 else 1

            yA = pd.Series(y[-q * AL :].values.reshape(-1, AL).mean(axis=1))

            xregA = self._prepare_xreg(xreg, AL, q, fhA) if xreg is not None else None

            AL_fit = self._fit_auto_ets(yA, ppyA)
            xregF = xregA[-fhA:] if xregA is not None else None

            FCs_temp = self.statetranslate(AL_fit, AL, fh, q, ppyA, 1, xreg=xregF)
            FCs[ALi, :, :] = FCs_temp

        combres = self.mapacomb(min(ALs), maximumAL, ppy, FCs, comb)
        forecasts = combres["forecasts"]

        if hybrid:
            forecasts = (FCs[0, 0, :] + forecasts) / 2

        return {"forecast": forecasts, "components": FCs}

    def _validate_xreg(self, xreg: pd.DataFrame, fh: int, observations: int):
        """Validate external regressor data."""
        if xreg.shape[0] < (fh + observations):
            raise ValueError("Number of observations in xreg must be >= len(y + fh).")

    def _prepare_xreg(
        self, xreg: pd.DataFrame, AL: int, q: int, fhA: int
    ) -> np.ndarray:
        """Prepare external regressor data for given aggregation level."""
        r = len(xreg) - q * AL
        p = xreg.shape[1]
        xregA = np.zeros((q + fhA, p))
        for k in range(p):
            temp = xreg.iloc[r : min((q + fhA) * AL, len(xreg)), k].values
            m = int(np.ceil(len(temp) / AL) * AL - len(temp))
            temp = np.pad(temp, (0, m), mode="constant", constant_values=np.nan)
            xregA[:, k] = np.nanmean(temp.reshape(-1, AL), axis=1)

        return xregA

    def _fit_exponential_smoothing(
        self, yA: np.ndarray, ppyA: int, ets_type: str
    ) -> ExponentialSmoothing:
        """Fit Exponential Smoothing model."""
        if ets_type == "ets":
            AL_fit = ExponentialSmoothing(trend="add", seasonal="add", sp=ppyA)
        elif ets_type == "es":
            AL_fit = ExponentialSmoothing(trend="add", seasonal="add", sp=ppyA)
        else:
            raise ValueError(f"Unknown ets_type: {ets_type}")
        return AL_fit.fit(pd.Series(yA))

    def _convert_ets_type(self, ets_char):
        if ets_char == "N":
            return None
        elif ets_char == "A":
            return "add"
        elif ets_char == "M":
            return "mul"
        else:
            return None

    def mapacomb(self, minimumAL, maximumAL, ppy, FCs, comb):
        """Combine the translated ets states."""
        k = maximumAL - minimumAL + 1
        fh = FCs.shape[2]

        perm_levels = np.zeros((k, fh))
        perm_seas = np.zeros((k, fh))

        for i in range(k):
            perm_levels[i, :] = FCs[i, 1, :] + FCs[i, 2, :]
            perm_seas[i, :] = FCs[i, 3, :]

        if comb == "w.mean":
            weights = np.arange(k, 0, -1)
            forecasts = np.average(FCs[:, 0, :], axis=0, weights=weights)
        elif comb == "w.median":
            weights = np.arange(k, 0, -1)
            forecasts = np.array(
                [self.weighted_median(FCs[:, 0, j], weights) for j in range(fh)]
            )
        elif comb == "mean":
            forecasts = np.mean(FCs[:, 0, :], axis=0)
        elif comb == "median":
            forecasts = np.median(FCs[:, 0, :], axis=0)
        else:  # "wght"
            weights = np.arange(k, 0, -1) / np.sum(np.arange(k, 0, -1))
            forecasts = np.sum(FCs[:, 0, :] * weights[:, np.newaxis], axis=0)

        return {
            "forecasts": forecasts,
            "perm_levels": perm_levels,
            "perm_seas": perm_seas,
        }

    def mapafor(
        self,
        y,
        mapafit,
        fh=-1,
        ifh=1,
        comb="w.mean",
        outplot=0,
        hybrid=True,
        conf_lvl=None,
        xreg=None,
    ):
        """MAPA in- and out-of-sample forecast."""
        observations = len(y)

        ALs = mapafit[mapafit["use"]]["AL"].values
        # minimumAL = min(ALs)
        maximumAL = max(ALs)
        ppy = mapafit["original.ppy"].iloc[0]
        ets_type = mapafit["etstype"].iloc[0]

        if fh == -1:
            fh = ppy

        ifh_c = max(ifh, fh) if conf_lvl is not None else ifh

        if ets_type == "ets":
            i_start = max(ppy, maximumAL)
        else:  # "es"
            i_start = max(ppy, maximumAL * 2)

        if ifh_c > 0 and i_start < observations:
            infor = np.full((ifh_c, observations), np.nan)
            for i in range(i_start, observations):
                inobs = y[:i]
                infor[:, i] = self.mapacalc(
                    inobs,
                    mapafit,
                    fh=ifh_c,
                    comb=comb,
                    outplot=0,
                    hybrid=hybrid,
                    xreg=xreg,
                )["forecast"]
                if i + ifh_c > observations:
                    k = i + ifh_c - observations
                    infor[(ifh_c - k) : ifh_c, i] = np.nan
        else:
            infor = None
            ifh_c = 0
            conf_lvl = None

        if fh > 0:
            outfor = self.mapacalc(
                y, mapafit, fh=fh, comb=comb, outplot=0, hybrid=hybrid, xreg=xreg
            )["forecast"]
        else:
            outfor = np.array([])

        if ifh_c == 1:
            resid = y - infor[0, :]
            MSE = np.nanmean(resid**2)
            MAE = np.nanmean(np.abs(resid))
        elif ifh_c > 1:
            MSE = np.full(ifh_c, np.nan)
            MAE = np.full(ifh_c, np.nan)
            for h in range(min(ifh_c, observations - ppy)):
                resid = y[h:] - infor[h, : (observations - h)]
                MSE[h] = np.nanmean(resid**2)
                MAE[h] = np.nanmean(np.abs(resid))
        else:
            MSE = None
            MAE = None

        PI = None
        if conf_lvl is not None:
            intv_idx = ~np.isnan(MSE)
            if np.sum(~intv_idx) > 0:
                intv = np.concatenate(
                    [
                        np.sqrt(MSE[intv_idx]),
                        np.sqrt(MSE[0])
                        * np.sqrt(np.arange(np.sum(intv_idx) + 1, fh + 1)),
                    ]
                )
            else:
                intv = np.sqrt(MSE[:fh])

            conf_lvl = np.minimum(conf_lvl, 0.999999999999999)
            conf_lvl = np.unique(conf_lvl)
            conf_lvl = np.sort(conf_lvl)[::-1]
            PIn = len(conf_lvl)
            z = norm.ppf(1 - (1 - conf_lvl) / 2)
            PI = np.zeros((2 * PIn, fh))
            for i in range(PIn):
                PI[i, :] = outfor + intv * z[i]
                PI[2 * PIn - i - 1, :] = outfor - intv * z[i]

        if ifh_c > 0 and ifh > 0:
            infor = infor[:ifh, :]
            MSE = MSE[:ifh]
            MAE = MAE[:ifh]
        else:
            infor = None
            MSE = None
            MAE = None

        if outplot == 1:
            self.plot_mapafor(y, infor, outfor, PI, observations, fh)

        return {"infor": infor, "outfor": outfor, "PI": PI, "MSE": MSE, "MAE": MAE}

    def weighted_median(self, data, weights):
        """Calculate weighted median."""
        sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
        cumsum = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumsum, sum(sorted_weights) / 2)
        return sorted_data[idx]

    def mapaest(
        self,
        y,
        ppy=None,
        minimum_al=1,
        maximum_al=None,
        paral=0,
        display=0,
        outplot=0,
        model="ZZZ",
        type="ets",
        xreg=None,
        pr_comp=0,
        **kwargs,
    ):
        """(MAPA) estimation function."""
        if ppy is None:
            if isinstance(y.index, pd.DatetimeIndex):
                ppy = y.index.freq.n
                if maximum_al is None:
                    maximum_al = ppy
            else:
                raise ValueError(
                    "Input ppy is not given and y input does not have a DatetimeIndex. "
                    "Please provide the periods in a season of the time series "
                    "at the sampled frequency."
                )

        if xreg is not None:
            raise NotImplementedError(
                "xreg functionality is not fully implemented yet."
            )

        if paral == 1:
            num_cores = multiprocessing.cpu_count()
            print(f"Running with {num_cores} cores")

        observations = len(y)

        mapafit = []
        if paral == 1:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                mapafit = list(
                    executor.map(
                        self.mapaest_loop,
                        range(1, maximum_al - minimum_al + 2),
                        [y] * (maximum_al - minimum_al + 1),
                        [minimum_al] * (maximum_al - minimum_al + 1),
                        [maximum_al] * (maximum_al - minimum_al + 1),
                        [observations] * (maximum_al - minimum_al + 1),
                        [ppy] * (maximum_al - minimum_al + 1),
                        [display] * (maximum_al - minimum_al + 1),
                        [model] * (maximum_al - minimum_al + 1),
                        [type] * (maximum_al - minimum_al + 1),
                        [xreg] * (maximum_al - minimum_al + 1),
                        [pr_comp] * (maximum_al - minimum_al + 1),
                        [kwargs] * (maximum_al - minimum_al + 1),
                    )
                )
        else:
            for i in range(1, maximum_al - minimum_al + 2):
                mapafit.append(
                    self.mapaest_loop(
                        i,
                        y,
                        minimum_al,
                        maximum_al,
                        observations,
                        ppy,
                        display,
                        model,
                        type,
                        xreg,
                        pr_comp,
                        **kwargs,
                    )
                )

        mapafit = pd.DataFrame(mapafit)
        mapafit.index = [f"AL{i}" for i in range(minimum_al, maximum_al + 1)]
        mapafit = mapafit.transpose()

        if outplot == 1:
            self.plot_mapa(mapafit)

        return mapafit

    def mapaest_loop(
        self,
        ali,
        y,
        minimum_al,
        maximum_al,
        observations,
        ppy,
        display,
        model,
        type,
        xreg,
        pr_comp,
        **kwargs,
    ):
        """Run a single loop in mapaest using the helper function."""
        al_vec = list(range(minimum_al, maximum_al + 1))
        al = al_vec[ali - 1]

        if display == 1:
            print(
                f"Aggregation level: {al}/{maximum_al} \
                ({round(100 * ali / (maximum_al - minimum_al + 1), 2)}%)"
            )

        q = observations // al
        ppy_a = ppy // al if ppy % al == 0 else 1

        y_a = y.groupby(y.index // al).mean()
        y_a.index = pd.date_range(
            start=y.index[0], periods=len(y_a), freq=f"{al}{y.index.freq.name}"
        )

        if q >= 4:
            try:
                fit = ExponentialSmoothing(
                    seasonal=model[2],
                    trend=model[1],
                    damped_trend=(len(model) == 4),
                    sp=ppy_a,
                ).fit(y_a, **kwargs)
                use = True
            except (ValueError, TypeError) as e:
                fit = None
                use = False
                print(f"An error occurred: {e}")
        else:
            fit = None
            use = False

        return {
            "model": fit,
            "AL": al,
            "original_ppy": ppy,
            "etstype": type,
            "use": use,
        }

    def plot_mapa(self, mapafit):
        """Plot function for MAPA results."""
        from sktime.utils.dependencies._dependencies import _check_soft_dependencies

        try:
            _check_soft_dependencies("matplotlib", severity="warning")
            import matplotlib.pyplot as plt
        except ImportError:
            # If matplotlib is not installed, we'll just return without plotting
            return
        als = mapafit.loc["AL"][mapafit.loc["use"]].astype(int)
        minimum_al = als.min()
        maximum_al = als.max()
        ppy = int(mapafit.loc["original_ppy"].iloc[0])

        perm_seas = [
            (ppy % al == 0) and (al < ppy) for al in range(minimum_al, maximum_al + 1)
        ]

        comps = np.zeros((len(als), 3))
        comps_char = np.empty((len(als), 3), dtype="U2")

        for i, al in enumerate(als):
            model = mapafit.loc["model"][mapafit.loc["AL"] == al].iloc[0]
            if model:
                components = [model.trend, model.damped_trend, model.seasonal]
                comps[i] = [1 if c else 0 for c in components]
                comps_char[i] = ["A" if c else "N" for c in components]
                if components[1]:
                    comps[i, 1] = 1.5
                    comps_char[i, 1] = "Ad"

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(comps.T, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Error", "Trend", "Season"][::-1])
        ax.set_xticks(range(len(als)))
        ax.set_xticklabels(als)
        ax.set_xlabel("Aggregation Level")
        ax.set_title("ETS components")

        for i in range(3):
            for j, al in enumerate(als):
                ax.text(j, i, comps_char[j, 2 - i], ha="center", va="center")

        for i, perm in enumerate(perm_seas):
            if not perm:
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, -0.5), 1, 3, fill=True, color="gray", alpha=0.5
                    )
                )

        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

    def plot_mapafor(self, y, infor, outfor, PI, observations, fh):
        """Plot function for MAPA forecasts."""
        from sktime.utils.dependencies._dependencies import _check_soft_dependencies

        try:
            _check_soft_dependencies("matplotlib", severity="warning")
            import matplotlib.pyplot as plt
        except ImportError:
            # If matplotlib is not installed, we'll just return without plotting
            return
        plt.figure(figsize=(12, 6))
        plt.plot(range(observations), y, label="Actual", color="black")

        if infor is not None:
            for i in range(infor.shape[0]):
                plt.plot(
                    range(observations - infor.shape[0] + i, observations),
                    infor[i, -infor.shape[0] + i :],
                    color="blue",
                    alpha=0.3,
                )

        if outfor is not None:
            plt.plot(
                range(observations - 1, observations + fh),
                np.concatenate([[y.iloc[-1]], outfor]),
                color="red",
                label="Forecast",
            )

        if PI is not None:
            plt.fill_between(
                range(observations - 1, observations + fh),
                np.concatenate([[y.iloc[-1]], PI[0, :]]),
                np.concatenate([[y.iloc[-1]], PI[-1, :]]),
                color="red",
                alpha=0.2,
                label="Prediction Interval",
            )

        plt.legend()
        plt.title("MAPA Forecast")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "comb": "w.mean",
            "hybrid": True,
            "conf_lvl": None,
            "sp": 12,
            "etstype": "ets",
        }
        params2 = {
            "comb": "median",
            "hybrid": False,
            "conf_lvl": 0.95,
            "sp": 7,
            "etstype": "es",
        }
        return [params1, params2]
