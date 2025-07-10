"""Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py."""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from skbase.utils.dependencies import _check_soft_dependencies

from .utils import _reduce

if _check_soft_dependencies(["torch"], severity="none"):
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from torch.nn.modules.loss import _Loss

    class sMAPELoss(_Loss):
        """sMAPELoss class."""

        __constants__ = ["reduction"]

        def __init__(
            self, size_average=None, reduce=None, reduction: str = "mean"
        ) -> None:
            super().__init__(size_average, reduce, reduction)

        def _abs(self, input):
            return F.l1_loss(input, torch.zeros_like(input), reduction="none")

        def _divide_no_nan(self, a: float, b: float) -> float:
            """Auxiliary function to handle divide by 0."""
            div = a / b
            div[div != div] = 0.0
            div[div == float("inf")] = 0.0
            return div

        def forward(self, input: Tensor, target: Tensor) -> Tensor:
            """Forward function."""
            delta_y = self._abs(input - target)
            scale = self._abs(target) + self._abs(input)
            error = self._divide_no_nan(delta_y, scale)
            error = 200 * torch.nanmean(error)

            return error

    def _divide_no_nan(a: float, b: float) -> float:
        """Auxiliary function to handle divide by 0."""
        div = a / b
        div[div != div] = 0.0
        div[div == float("inf")] = 0.0
        return div

    @dataclass
    class ForecastingMetrics:
        """Forecasting Metrics."""

        mae: Union[float, np.ndarray] = None
        mse: Union[float, np.ndarray] = None
        mape: Union[float, np.ndarray] = None
        smape: Union[float, np.ndarray] = None
        rmse: Union[float, np.ndarray] = None

    def mae(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        r"""Calculate MAE.

        Calculates Mean Absolute Error (MAE) between
        y and y_hat. MAE measures the relative prediction
        accuracy of a forecasting method by calculating the
        deviation of the prediction and the true
        value at a given time and averages these devations
        over the length of the series.

        $$ \\mathrm{MAE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) =
            \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1}
            |y_{\\tau} - \\hat{y}_{\\tau}| $$

        Parameters
        ----------
            y: numpy array.
                Observed values.
            y_hat: numpy array
                Predicted values.
            reduction: str, optional.
                Type of reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output,
                'sum': the output will be summed.
            axis: None or int, optional.
                Axis or axes along which to average a.
                The default, axis=None, will average over all of the elements of
                the input array. If axis is negative it counts from last to first.

        Returns
        -------
            mae: numpy array or double.
                Return the MAE along the specified axis.
        """
        delta_y = np.abs(y - y_hat)
        return _reduce(delta_y, reduction=reduction, axis=axis)

    def mse(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        r"""Calculate MSE.

        Calculates Mean Squared Error (MSE) between
        y and y_hat. MSE measures the relative prediction
        accuracy of a forecasting method by calculating the
        squared deviation of the prediction and the true
        value at a given time, and averages these devations
        over the length of the series.

        $$ \\mathrm{MSE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) =
            \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} -
            \\hat{y}_{\\tau})^{2} $$

        Parameters
        ----------
            y: numpy array.
                Actual test values.
            y_hat: numpy array.
                Predicted values.
            reduction: str, optional.
                Type of reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output,
                'sum': the output will be summed.
            axis: None or int, optional.
                Axis or axes along which to average a.
                The default, axis=None, will average over all of the
                elements of the input array. If axis is negative it counts
                from the last to the first axis.

        Returns
        -------
            mse: numpy array or double.
                Return the MSE along the specified axis.
        """
        delta_y = np.square(y - y_hat)
        return _reduce(delta_y, reduction=reduction, axis=axis)

    def rmse(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        r"""Calculate Rmse.

        Calculates Root Mean Squared Error (RMSE) between
        y and y_hat. RMSE measures the relative prediction
        accuracy of a forecasting method by calculating the squared deviation
        of the prediction and the observed value at a given time and
        averages these devations over the length of the series.
        Finally the RMSE will be in the same scale
        as the original time series so its comparison with other
        series is possible only if they share a common scale.
        RMSE has a direct connection to the L2 norm.

        $$ \\mathrm{RMSE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) =
            \\sqrt{\\frac{1}{H} \\sum^{t+H}_{\\tau=t+1} (y_{\\tau} -
            \\hat{y}_{\\tau})^{2}} $$

        Parameters
        ----------
            y: numpy array.
                Observed values.
            y_hat: numpy array.
                Predicted values.
            reduction: str, optional.
                Type of reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output,
                'sum': the output will be summed.
            axis: None or int, optional.
                Axis or axes along which to average a.
                The default, axis=None, will average over all of the elements of
                the input array. If axis is negative it counts from the last to
                first.

        Returns
        -------
            rmse: numpy array or double.
                Return the RMSE along the specified axis.
        """
        return np.sqrt(mse(y, y_hat, reduction, axis))

    def mape(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        r"""Calculate mape.

        Calculates Mean Absolute Percentage Error (MAPE) between
        y and y_hat. MAPE measures the relative prediction
        accuracy of a forecasting method by calculating the percentual deviation
        of the prediction and the observed value at a given time and
        averages these devations over the length of the series.
        The closer to zero an observed value is, the higher penalty MAPE loss
        assigns to the corresponding error.

        $$ \\mathrm{MAPE}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) =
            \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1}
            \\frac{|y_{\\tau}-\\hat{y}_{\\tau}|}{|y_{\\tau}|} $$

        Parameters
        ----------
            y: numpy array.
                Observed values.
            y_hat: numpy array.
                Predicted values.
            reduction: str, optional.
                Type of reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements
                in the output,
                'sum': the output will be summed.
            axis: None or int, optional.
                Axis or axes along which to average a.
                The default, axis=None, will average over all of the elements of
                the input array. If axis is negative it counts from the last to
                first.

        Returns
        -------
            mape: numpy array or double.
                Return the MAPE along the specified axis.
        """
        delta_y = np.abs(y - y_hat)
        scale = np.abs(y)
        error = _divide_no_nan(delta_y, scale)
        return 100 * _reduce(error, reduction=reduction, axis=axis)

    def smape(
        y: np.ndarray,
        y_hat: np.ndarray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        r"""Calculate smape.

        Calculates Symmetric Mean Absolute Percentage Error (SMAPE) between
        y and y_hat. SMAPE measures the relative prediction
        accuracy of a forecasting method by calculating the relative deviation
        of the prediction and the observed value scaled by the sum of the
        absolute values for the prediction and observed value at a
        given time, then averages these devations over the length
        of the series. This allows the SMAPE to have bounds between
        0% and 200% which is desireble compared to normal MAPE that
        may be undetermined when the target is zero.

        $$ \\mathrm{SMAPE}_{2}(\\mathbf{y}_{\\tau}, \\mathbf{\\hat{y}}_{\\tau}) =
        \\frac{1}{H} \\sum^{t+H}_{\\tau=t+1}
        \\frac{|y_{\\tau}-\\hat{y}_{\\tau}|}{|y_{\\tau}|+|\\hat{y}_{\\tau}|} $$

        Parameters
        ----------
            y: numpy array.
                Observed values.
            y_hat: numpy array.
                Predicted values.
            reduction: str, optional.
                Type of reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements
                in the output,
                'sum': the output will be summed.
            axis: None or int, optional.
                Axis or axes along which to average a.
                The default, axis=None, will average over all of the elements of
                the input array. If axis is negative it counts from the last to
                first.

        Returns
        -------
            smape: numpy array or double.
                Return the SMAPE along the specified axis.
        """
        delta_y = np.abs(y - y_hat)
        scale = np.abs(y) + np.abs(y_hat)
        error = _divide_no_nan(delta_y, scale)
        error = 200 * _reduce(error, reduction=reduction, axis=axis)

        if isinstance(error, float):
            # assert error <= 200, "SMAPE should be lower than 200"
            if error > 200:
                warnings.warn(
                    f"SMAPE should be lower than 200 but was found to be {error}"
                )
        else:
            # assert all(error <= 200), "SMAPE should be lower than 200"
            if all(error > 200):
                warnings.warn(
                    f"SMAPE should be lower than 200 but was found to be {error}"
                )

        return error

    def get_forecasting_metrics(
        y: npt.NDArray,
        y_hat: npt.NDArray,
        reduction: str = "mean",
        axis: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """Get Forecasting Metrics."""
        return ForecastingMetrics(
            mae=mae(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
            mse=mse(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
            mape=mape(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
            smape=smape(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
            rmse=rmse(y=y, y_hat=y_hat, axis=axis, reduction=reduction),
        )
else:

    class sMAPELoss:
        """Create dummy class if torch is unavailable."""

        pass

    class ForecastingMetrics:
        """Create dummy class torch is unavailable."""

        pass

    def mae():
        """Create dummy function if torch is not available."""
        pass

    def mse():
        """Create dummy function if torch is not available."""
        pass

    def rmse():
        """Create dummy function if torch is not available."""
        pass

    def mape():
        """Create dummy function if torch is not available."""
        pass

    def smape():
        """Create dummy function if torch is not available."""
        pass

    def get_forecasting_metrics():
        """Create dummy function if torch is not available."""
        pass
