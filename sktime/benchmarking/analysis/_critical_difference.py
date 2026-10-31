"""Critical-difference diagram evaluator for benchmark results."""

__all__ = ["CriticalDifferenceDiagram"]

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer


class CriticalDifferenceDiagram(BaseBenchmarkAnalyzer):
    """Critical-difference (CD) diagram evaluator.

    Object-oriented wrapper around
    ``sktime.benchmarking.critical_difference.plot_critical_difference``. It
    computes average ranks and statistically similar cliques (Friedman +
    Nemenyi) and renders the Demsar critical-difference diagram.

    The diagram is the intended rendering primitive for the future Hugging Face
    leaderboard; ``plot`` returns a ``(fig, ax)`` tuple so the figure can be
    embedded without rendering side effects.

    Parameters
    ----------
    metric : str, optional (default=None)
        Metric to analyse; see ``BaseBenchmarkAnalyzer``.
    lower_is_better : bool, optional (default=True)
        Whether lower scores are better; forwarded as ``is_errors`` to the
        underlying diagram.
    alpha : float, optional (default=0.05)
        Significance level. One of ``0.01``, ``0.05`` or ``0.1``.
    width : int, optional (default=10)
        Figure width in inches.
    textspace : float, optional (default=2.5)
        Space (in inches) reserved on the figure sides for method names.
    reverse : bool, optional (default=True)
        If ``True``, the lowest rank is drawn on the right.
    """

    _tags = {
        "python_dependencies": "matplotlib",
        "property:analyzer_type": "plot",
    }

    def __init__(
        self,
        metric=None,
        lower_is_better=True,
        alpha=0.05,
        width=10,
        textspace=2.5,
        reverse=True,
    ):
        self.alpha = alpha
        self.width = width
        self.textspace = textspace
        self.reverse = reverse
        super().__init__(metric=metric, lower_is_better=lower_is_better)

    def _evaluate(self, scores):
        """Return the average ranks underlying the diagram.

        Returns
        -------
        pandas.DataFrame
            Columns ``["model_id", "rank"]``, sorted by ascending rank.
        """
        return self._mean_ranks(scores)

    def plot(self, results):
        """Render the critical-difference diagram.

        Parameters
        ----------
        results : pandas.DataFrame or str or pathlib.Path
            Benchmark results, as accepted by ``evaluate``.

        Returns
        -------
        (fig, ax) : tuple of matplotlib Figure and Axes
        """
        _check_soft_dependencies("matplotlib")

        from sktime.benchmarking.critical_difference import plot_critical_difference

        scores = self._coerce_to_score_matrix(results)
        return plot_critical_difference(
            scores.to_numpy(),
            list(scores.columns),
            is_errors=self.lower_is_better,
            alpha=self.alpha,
            width=self.width,
            textspace=self.textspace,
            reverse=self.reverse,
            return_fig=True,
        )
