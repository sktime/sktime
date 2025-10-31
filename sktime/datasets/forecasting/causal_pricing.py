"""Synthetic causal pricing dataset generator for forecasting."""

__author__ = ["XAheli", "geetu040"]

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state

from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["CausalPricing", "make_causal_pricing"]

# Constants from paper
DISCOUNT_LEVELS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # Eq 29: d(j_t) = j_t * 0.1
TARGET_AVG_DISCOUNT = 0.14  # Eq 25: Initialize stock to clear at 14% avg discount
PROMOTION_PROBABILITY = 0.3  # Promotion is noise variable with 30% probability


def make_causal_pricing(
    n_series=4467,
    n_timepoints=100,
    n_categories_d=45,
    n_categories_k=15,
    random_state=None,
    return_ground_truth=True,
):
    """Generate synthetic retail pricing dataset with known causal structure.

    Creates Panel scitype time series data simulating retail demand with:
    - Known treatment effects (price elasticity per article)
    - Dynamic pricing policy with feedback loops
    - Realistic seasonality, trends, and categorical structure
    - Confounding between price and demand

    Implements data generation from Schultz et al. (2024), Appendix E.

    Parameters
    ----------
    n_series : int, default=4467
        Number of articles (time series instances)
    n_timepoints : int, default=100
        Number of time periods (weeks)
    n_categories_d : int, default=45
        Number of brand categories
    n_categories_k : int, default=15
        Number of product group categories
    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility
    return_ground_truth : bool, default=True
        If True, return (X, y, ground_truth) tuple
        If False, return (X, y) tuple

    Returns
    -------
    X : pd.DataFrame with MultiIndex
        Features in Panel scitype format with index levels:
        - article_id : int, article identifier
        - time : int, time period (0 to n_timepoints-1)

        Columns:
        - discount : float, discount rate (0.0-0.5)
        - stock : float, inventory level
        - week_number : int, time index
        - d : int, category d (brand)
        - k : int, category k (product group)
        - promotion : int, promotion flag (noise, no causal effect)
        - p0 : float, initial price

    y : pd.DataFrame with MultiIndex
        Target variable with same index as X
        Column:
        - demand : float, observed demand

    ground_truth : dict (only if return_ground_truth=True)
        Dictionary containing:
        - treatment_effects : pd.Series, article-specific price elasticity
        - base_demand : pd.DataFrame, demand without price effects
        - causal_dag : pgmpy.base.DAG, causal graph structure (requires pgmpy)
        - metadata : dict, generation parameters

    Raises
    ------
    ValueError
        If n_series <= 0, n_timepoints < 5, n_categories_d <= 0, or n_categories_k <= 0
    UserWarning
        If insufficient valid articles generated after maximum attempts
    ImportError
        If return_ground_truth=True and pgmpy is not installed

    Notes
    -----
    The causal DAG (directed acyclic graph) is a pgmpy.base.DAG object,
    which is built on networkx.DiGraph. It represents the causal structure:

    - Nodes: Variables in the causal system (demand, price, stock, etc.)
    - Edges: Directed causal relationships (cause -> effect)
    - Key causal path: stock -> discount -> price -> demand

    The DAG provides methods like:
    - dag.nodes() - list all variables
    - dag.edges() - list all causal relationships
    - dag.has_edge(u, v) - check if u causes v
    - dag.get_parents(node) - get direct causes
    - dag.get_children(node) - get direct effects

    References
    ----------
    .. [1] Schultz et al. (2024), "Causal Forecasting for Pricing",
       arXiv:2312.15282, https://arxiv.org/abs/2312.15282

    Examples
    --------
    >>> from sktime.datasets.forecasting import make_causal_pricing
    >>> X, y = make_causal_pricing(
    ...     n_series=100, n_timepoints=50, return_ground_truth=False, random_state=42
    ... )  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (5000, 7)
    >>> y.shape  # doctest: +SKIP
    (5000, 1)
    """
    if n_series <= 0:
        raise ValueError(f"n_series must be positive, got {n_series}")
    if n_timepoints < 5:
        raise ValueError(
            f"n_timepoints must be >= 5 for pricing policy to work, got {n_timepoints}"
        )
    if n_categories_d <= 0:
        raise ValueError(f"n_categories_d must be positive, got {n_categories_d}")
    if n_categories_k <= 0:
        raise ValueError(f"n_categories_k must be positive, got {n_categories_k}")

    rng = check_random_state(random_state)

    max_attempts = 10
    n_generate = int(n_series * 1.5)

    for attempt in range(max_attempts):
        category_assignments = _generate_categories(
            n_generate, n_categories_d, n_categories_k, rng
        )

        base_demand, components = _generate_base_demand(
            n_generate, n_timepoints, category_assignments, rng
        )

        treatment_effects = _generate_treatment_effects(
            n_generate, components["a_it"], rng
        )

        initial_prices = _generate_initial_prices(n_generate, base_demand, rng)

        initial_stock = _generate_initial_stock(
            n_generate,
            n_timepoints,
            base_demand,
            initial_prices,
            treatment_effects,
            rng,
        )

        prices, discounts, stock, demand = _simulate_pricing_policy(
            n_generate,
            n_timepoints,
            base_demand,
            treatment_effects,
            initial_prices,
            initial_stock,
            rng,
        )

        valid_articles = _filter_valid_articles(demand, stock)

        if len(valid_articles) >= n_series:
            valid_articles = valid_articles[:n_series]
            break

        n_generate = int(n_generate * 1.5)
    else:
        import warnings

        warnings.warn(
            f"Could not generate {n_series} valid articles after "
            f"{max_attempts} attempts. Returning {len(valid_articles)} articles. "
            f"Try: reducing n_categories_d/k, increasing n_timepoints, "
            f"or different random_state.",
            UserWarning,
            stacklevel=2,
        )

    category_assignments = {
        "d": category_assignments["d"][valid_articles],
        "k": category_assignments["k"][valid_articles],
    }

    X, y = _format_as_panel(
        valid_articles,
        n_timepoints,
        prices,
        discounts,
        stock,
        demand,
        category_assignments,
        initial_prices,
        rng,
    )

    if return_ground_truth:
        ground_truth = _prepare_ground_truth(
            valid_articles,
            n_timepoints,
            treatment_effects,
            base_demand,
            n_categories_d,
            n_categories_k,
            random_state,
        )
        return X, y, ground_truth
    else:
        return X, y


def _generate_categories(n_series, n_categories_d, n_categories_k, rng):
    """Assign articles to categories d and k."""
    return {
        "d": rng.randint(0, n_categories_d, size=n_series),
        "k": rng.randint(0, n_categories_k, size=n_series),
    }


def _generate_base_demand(n_series, n_timepoints, category_assignments, rng):
    """Generate base demand components (Equations 14-24)."""
    d_assignments = category_assignments["d"]
    k_assignments = category_assignments["k"]

    n_categories_d = d_assignments.max() + 1
    n_categories_k = k_assignments.max() + 1

    alpha_d = rng.normal(10, 3, size=n_categories_d)
    beta_k = rng.normal(300, 50, size=n_categories_k)

    epsilon_it = rng.normal(0, 1, size=(n_series, n_timepoints))
    psi_it = rng.normal(0, 5, size=(n_series, n_timepoints))

    a_it = alpha_d[d_assignments][:, np.newaxis] + epsilon_it
    b_it = beta_k[k_assignments][:, np.newaxis] + psi_it

    c_it = 0.05 * a_it**2 + 0.25 * a_it + 0.5 * b_it

    gamma_i = rng.uniform(-0.02, 0.02, size=n_series)
    sigma_tau_i = rng.uniform(0, 0.15, size=n_series)

    t_array = np.arange(n_timepoints)
    tau_it = gamma_i[:, np.newaxis] * t_array + rng.normal(
        0, sigma_tau_i[:, np.newaxis], size=(n_series, n_timepoints)
    )

    n_season_types = 6
    season_assignments = k_assignments % n_season_types
    season_shifts = rng.randint(-15, 16, size=n_season_types)

    s_it = np.zeros((n_series, n_timepoints))
    for i in range(n_series):
        shift = season_shifts[season_assignments[i]]
        s_it[i, :] = np.sin(2 * np.pi * (t_array + shift) / 30)

    base_demand = (0.15 * tau_it + 0.25 * s_it + 1) * c_it

    components = {
        "a_it": a_it,
        "b_it": b_it,
        "c_it": c_it,
        "tau_it": tau_it,
        "s_it": s_it,
    }

    return base_demand, components


def _generate_treatment_effects(n_series, a_it, rng):
    """Generate article-specific treatment effects (Equation 20)."""
    e_base = np.maximum(1.3, rng.lognormal(0.75, 0.125, size=n_series))

    a_bar = np.mean(a_it, axis=1)

    treatment_effects = -e_base * 0.15 * a_bar

    return treatment_effects


def _generate_initial_prices(n_series, base_demand, rng):
    """Generate initial prices (Equation 21)."""
    q_bar = np.mean(base_demand, axis=1)

    initial_prices = np.zeros(n_series)
    for i in range(n_series):
        mean_price = q_bar[i] / 3
        std_price = q_bar[i] / 1.5

        price = rng.normal(mean_price, std_price)
        while price <= 0:
            price = rng.normal(mean_price, std_price)

        initial_prices[i] = price

    return initial_prices


def _generate_initial_stock(
    n_series, n_timepoints, base_demand, initial_prices, treatment_effects, rng
):
    """Generate initial stock levels (Equation 25 with interpretation).

    Note: Paper's Equation 25 literal interpretation gives negative stock
    when treatment effects are negative. We instead interpret as sum of
    expected demand at TARGET_AVG_DISCOUNT (14%) average discount:
        z_0 = sum_t[q^b_{i,t} + p_{i,0} * (1-0.14) * e_i]

    This ensures physically valid (positive) stock levels while maintaining
    the intent of initializing stock to clear at ~14% average discount.
    """
    initial_stock = np.zeros(n_series)

    for i in range(n_series):
        expected_demand = np.sum(
            base_demand[i, :]
            + initial_prices[i] * (1 - TARGET_AVG_DISCOUNT) * treatment_effects[i]
        )
        initial_stock[i] = max(0, expected_demand)

    return initial_stock


def _simulate_pricing_policy(
    n_series,
    n_timepoints,
    base_demand,
    treatment_effects,
    initial_prices,
    initial_stock,
    rng,
):
    """Simulate dynamic pricing policy (Equations 26-30).

    Note: Implements reactive pricing where discounts are adjusted based on
    stock coverage to clear inventory by end of horizon.
    """
    prices = np.zeros((n_series, n_timepoints))
    discounts = np.zeros((n_series, n_timepoints))
    stock = np.zeros((n_series, n_timepoints))
    demand = np.zeros((n_series, n_timepoints))

    for i in range(n_series):
        stock[i, 0] = initial_stock[i]
        prices[i, 0:4] = initial_prices[i]
        discounts[i, 0:4] = 0.0

        j_t = 0

        for t in range(n_timepoints):
            if t >= 4:
                recent_demand_sum = demand[i, max(0, t - 3) : t].sum()
                if recent_demand_sum > 0:
                    m_t = 4 * stock[i, t] / recent_demand_sum
                else:
                    m_t = n_timepoints - t

                w_t = m_t / (n_timepoints - t) if (n_timepoints - t) > 0 else 1.0

                lambda_it = rng.uniform()

                if w_t > 1 and lambda_it > (1 / w_t):
                    j_t = min(j_t + 1, len(DISCOUNT_LEVELS) - 1)
                elif w_t < 1 and lambda_it > w_t:
                    j_t = max(j_t - 1, 0)

                discounts[i, t] = DISCOUNT_LEVELS[j_t]
                prices[i, t] = initial_prices[i] * (1 - discounts[i, t])

            demand_it = base_demand[i, t] + prices[i, t] * treatment_effects[i]
            demand[i, t] = max(0, demand_it)

            if t < n_timepoints - 1:
                stock[i, t + 1] = max(0, stock[i, t] - demand[i, t])

    return prices, discounts, stock, demand


def _filter_valid_articles(demand, stock):
    """Filter articles with negative demand or stock issues."""
    valid = (demand >= 0).all(axis=1) & (stock[:, 0] > 0)
    return np.where(valid)[0]


def _format_as_panel(
    valid_articles,
    n_timepoints,
    prices,
    discounts,
    stock,
    demand,
    category_assignments,
    initial_prices,
    rng,
):
    """Format data as Panel scitype with pd.MultiIndex."""
    n_valid = len(valid_articles)

    promotion = rng.binomial(1, PROMOTION_PROBABILITY, size=n_valid)

    data_dict = {
        "discount": [],
        "stock": [],
        "week_number": [],
        "d": [],
        "k": [],
        "promotion": [],
        "p0": [],
    }

    demand_list = []

    article_ids = []
    time_ids = []

    for idx, i in enumerate(valid_articles):
        for t in range(n_timepoints):
            article_ids.append(idx)
            time_ids.append(t)

            data_dict["discount"].append(discounts[i, t])
            data_dict["stock"].append(stock[i, t])
            data_dict["week_number"].append(t)
            data_dict["d"].append(category_assignments["d"][idx])
            data_dict["k"].append(category_assignments["k"][idx])
            data_dict["promotion"].append(promotion[idx])
            data_dict["p0"].append(initial_prices[i])

            demand_list.append(demand[i, t])

    index = pd.MultiIndex.from_arrays(
        [article_ids, time_ids], names=["article_id", "time"]
    )

    X = pd.DataFrame(data_dict, index=index)
    y = pd.DataFrame({"demand": demand_list}, index=index)

    return X, y


def _prepare_ground_truth(
    valid_articles,
    n_timepoints,
    treatment_effects,
    base_demand,
    n_categories_d,
    n_categories_k,
    random_state,
):
    """Prepare ground truth dictionary with pgmpy causal DAG.

    DAG is a networkx-based directed acyclic graph from pgmpy.base.
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("pgmpy", severity="error", obj="make_causal_pricing")

    treatment_effects_valid = pd.Series(
        treatment_effects[valid_articles],
        index=pd.Index(range(len(valid_articles)), name="article_id"),
        name="treatment_effect",
    )

    base_demand_data = []
    article_ids = []
    time_ids = []

    for idx, i in enumerate(valid_articles):
        for t in range(n_timepoints):
            article_ids.append(idx)
            time_ids.append(t)
            base_demand_data.append(base_demand[i, t])

    base_demand_df = pd.DataFrame(
        {"base_demand": base_demand_data},
        index=pd.MultiIndex.from_arrays(
            [article_ids, time_ids], names=["article_id", "time"]
        ),
    )

    from pgmpy.base import DAG

    causal_dag = DAG()

    # Add nodes representing variables in the causal system
    causal_dag.add_nodes_from(
        [
            "base_demand",
            "price",
            "discount",
            "stock",
            "demand",
            "category_d",
            "category_k",
            "seasonality",
            "trend",
        ]
    )

    # Add directed edges representing causal relationships
    # Format: (cause, effect)
    causal_dag.add_edges_from(
        [
            ("base_demand", "demand"),  # Base demand causes observed demand
            ("price", "demand"),  # Price causes demand (treatment effect)
            ("stock", "discount"),  # Stock level causes discount decision
            ("discount", "price"),  # Discount causes effective price
            ("category_d", "base_demand"),  # Category d affects base demand
            ("category_k", "base_demand"),  # Category k affects base demand
            ("category_k", "seasonality"),  # Category k determines seasonality type
            ("seasonality", "base_demand"),  # Seasonality affects base demand
            ("trend", "base_demand"),  # Trend affects base demand
        ]
    )

    ground_truth = {
        "treatment_effects": treatment_effects_valid,
        "base_demand": base_demand_df,
        "causal_dag": causal_dag,
        "metadata": {
            "n_series": len(valid_articles),
            "n_timepoints": n_timepoints,
            "n_categories_d": n_categories_d,
            "n_categories_k": n_categories_k,
            "reference": "Schultz et al. (2024), arXiv:2312.15282",
            "random_state": random_state,
            "causal_structure": "q_it = q^b_it + p_it * e_i",
            "treatment_variable": "price",
            "outcome_variable": "demand",
            "heterogeneous_effects": True,
        },
    }

    return ground_truth


class CausalPricing(_ForecastingDatasetFromLoader):
    """Generate synthetic retail pricing dataset with known causal structure.

    Creates Panel scitype time series data simulating retail demand with:
    - Known treatment effects (price elasticity per article)
    - Dynamic pricing policy with feedback loops
    - Realistic seasonality, trends, and categorical structure
    - Confounding between price and demand

    Parameters
    ----------
    n_series : int, default=4467
        Number of articles (time series instances)
    n_timepoints : int, default=100
        Number of time periods (weeks)
    n_categories_d : int, default=45
        Number of brand categories
    n_categories_k : int, default=15
        Number of product group categories
    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility
    return_ground_truth : bool, default=True
        If True, return (X, y, ground_truth) tuple
        If False, return (X, y) tuple

    Examples
    --------
    >>> from sktime.datasets.forecasting import CausalPricing
    >>> loader = CausalPricing(n_series=100, n_timepoints=50)
    >>> X, y = loader.load("X", "y")  # doctest: +SKIP
    >>> X.shape  # doctest: +SKIP
    (5000, 7)
    >>> y.shape  # doctest: +SKIP
    (5000, 1)

    Notes
    -----
    This synthetic dataset simulates retail pricing dynamics with:

    - Panel structure: Multiple articles tracked over time
    - Known causal relationships: Price affects demand through
      article specific elasticity
    - Dynamic pricing: Discount decisions based on stock levels
    - Realistic features: Seasonality, trends, categorical effects
    - Ground truth: Optional return of true causal effects and DAG structure

    The data generation implements equations 13-30 from Schultz et al. (2024)
    Appendix E, with two documented deviations:
    1. Stock initialization (Eq 25): Modified to ensure positive stock levels
    2. Treatment effect sign: Negative elasticity (normal goods)

    Dimensionality:     Panel with 7 features
    Default series:     4467 articles
    Default length:     100 weeks
    Frequency:          Weekly
    Scitype:            Panel

    References
    ----------
    .. [1] Schultz et al. (2024), "Causal Forecasting for Pricing",
       arXiv:2312.15282, https://arxiv.org/abs/2312.15282
    """

    _tags = {
        "name": "causal_pricing",
        "python_dependencies": [
            "pgmpy"
        ],  # Required for return_ground_truth=True (default)
        "n_splits": 0,
        "is_univariate": True,  # y (demand) is univariate
        "is_one_series": False,  # Multiple series (panel of articles)
        "is_one_panel": True,  # All series in one panel
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": True,  # X contains exogenous features
        "n_instances": 4467,  # Default n_series (number of articles)
        "n_timepoints": 100,  # Default n_timepoints
        "n_timepoints_train": 0,
        "n_timepoints_test": 0,
        "frequency": "W",  # Weekly
        "n_dimensions": 1,  # y is univariate (demand)
        "n_panels": 1,  # One panel containing multiple series
        "n_hierarchy_levels": 0,
    }

    loader_func = make_causal_pricing

    def __init__(
        self,
        n_series=4467,
        n_timepoints=100,
        n_categories_d=45,
        n_categories_k=15,
        random_state=None,
        return_ground_truth=True,
    ):
        """Initialize CausalPricing dataset.

        Parameters
        ----------
        n_series : int, default=4467
            Number of articles (time series instances)
        n_timepoints : int, default=100
            Number of time periods (weeks)
        n_categories_d : int, default=45
            Number of brand categories
        n_categories_k : int, default=15
            Number of product group categories
        random_state : int, RandomState instance or None, default=None
            Random seed for reproducibility
        return_ground_truth : bool, default=True
            If True, return (X, y, ground_truth) tuple
            If False, return (X, y) tuple
        """
        self.n_series = n_series
        self.n_timepoints = n_timepoints
        self.n_categories_d = n_categories_d
        self.n_categories_k = n_categories_k
        self.random_state = random_state
        self.return_ground_truth = return_ground_truth

        # Update dynamic tags based on parameters
        self._tags["n_instances"] = n_series
        self._tags["n_timepoints"] = n_timepoints
        self._tags["n_panels"] = n_series

        super().__init__()

    def _split_into_y_and_X(self, loader_output):
        """Split the output of the loader into X and y.

        Handles the optional ground_truth return value.

        Parameters
        ----------
        loader_output : tuple
            Output of make_causal_pricing function.

        Returns
        -------
        tuple
            Tuple containing (y, X).
        """
        if self.return_ground_truth:
            X, y, ground_truth = loader_output
            # Store ground_truth as instance attribute for later access
            self.ground_truth_ = ground_truth
        else:
            X, y = loader_output

        return y, X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the dataset.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class
        """
        # Minimal parameters with ground truth
        params1 = {
            "n_series": 5,
            "n_timepoints": 10,
            "random_state": 42,
            "return_ground_truth": True,
        }

        # Custom categories with ground truth
        params2 = {
            "n_series": 10,
            "n_timepoints": 20,
            "n_categories_d": 5,
            "n_categories_k": 3,
            "random_state": 123,
            "return_ground_truth": True,
        }

        params3 = {
            "n_series": 8,
            "n_timepoints": 15,
            "n_categories_d": 10,
            "n_categories_k": 5,
            "random_state": 999,
            "return_ground_truth": True,
        }

        return [params1, params2, params3]
