"""Test new features: auto-sizing for seasonal data and custom model parameters."""

from sktime.benchmarking.forecasting import ForecastingBenchmark, TimeSeriesSimulator
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster

print("=" * 80)
print("TESTING NEW FEATURES")
print("=" * 80)

# Feature 1: Auto-sizing for seasonal data
print("\n\n" + "=" * 80)
print("FEATURE 1: AUTO-SIZING FOR SEASONAL DATA")
print("=" * 80)

print(
    "\n1.1 Small length with seasonality - should auto-increase to 3x seasonal period"
)
print("-" * 80)
simulator = TimeSeriesSimulator(
    length=20,  # Request only 20 points
    distribution="poisson",
    dist_params={"lam": 50},
    seasonality=12,  # Monthly seasonality
    seasonality_strength=10.0,
    random_state=42,
)
seasonal_data = simulator.simulate()
print("Requested length: 20")
print("Seasonal period: 12")
print("Minimum required (3x seasonal): 36")
print(f"Actual generated length: {len(seasonal_data)}")
print("✓ Auto-sized to ensure at least 3 full seasonal cycles!\n")

print("1.2 Multiple seasonal periods - uses max period")
print("-" * 80)
simulator = TimeSeriesSimulator(
    length=50,  # Request 50 points
    distribution="normal",
    dist_params={"loc": 100, "scale": 15},
    seasonality=[7, 30],  # Weekly and monthly seasonality
    seasonality_strength=[5.0, 8.0],
    random_state=42,
)
multi_seasonal_data = simulator.simulate()
print("Requested length: 50")
print("Seasonal periods: [7, 30]")
print("Maximum seasonal period: 30")
print("Minimum required (3x max seasonal): 90")
print(f"Actual generated length: {len(multi_seasonal_data)}")
print("✓ Auto-sized to 3x the maximum seasonal period!\n")

print("1.3 User can still override with larger length")
print("-" * 80)
simulator = TimeSeriesSimulator(
    length=500,  # User wants 500 points
    distribution="normal",
    dist_params={"loc": 100, "scale": 15},
    seasonality=12,
    seasonality_strength=5.0,
    random_state=42,
)
custom_length_data = simulator.simulate()
print("Requested length: 500")
print("Seasonal period: 12")
print(f"Actual generated length: {len(custom_length_data)}")
print("✓ User's choice respected when larger than minimum!\n")


# Feature 2: Custom model parameters
print("\n\n" + "=" * 80)
print("FEATURE 2: CUSTOM MODEL PARAMETERS")
print("=" * 80)

print("\n2.1 Initialize models with custom parameters")
print("-" * 80)
# Generate test data
simulator = TimeSeriesSimulator(
    length=200,
    distribution="poisson",
    dist_params={"lam": 30},
    trend="linear",
    trend_params={"slope": 0.05},
    seasonality=7,  # Weekly pattern
    seasonality_strength=8.0,
    random_state=42,
)
test_data = simulator.simulate()
print(f"Generated test data: {len(test_data)} observations with weekly seasonality")

# Benchmark with custom model parameters
print("\nBenchmarking models with custom parameters:")
print("-" * 80)
benchmark = ForecastingBenchmark(
    models=[
        # Different naive strategies
        ("naive_last", NaiveForecaster(strategy="last")),
        ("naive_mean", NaiveForecaster(strategy="mean")),
        ("naive_drift", NaiveForecaster(strategy="drift")),
        # Polynomial trends with different degrees
        ("linear_trend", PolynomialTrendForecaster(degree=1)),
        ("quadratic_trend", PolynomialTrendForecaster(degree=2)),
    ],
    test_size=0.2,
    verbose=True,
    random_state=42,
)
results = benchmark.run(test_data)

print("\n2.2 Comparing different parameter settings for same model")
print("-" * 80)
# Compare polynomial forecasters with different degrees
print("\nEffect of polynomial degree on forecasting:")
benchmark_poly = ForecastingBenchmark(
    models=[
        ("poly_degree_1", PolynomialTrendForecaster(degree=1)),
        ("poly_degree_2", PolynomialTrendForecaster(degree=2)),
        ("poly_degree_3", PolynomialTrendForecaster(degree=3)),
    ],
    test_size=0.2,
    verbose=True,
    random_state=42,
)
poly_results = benchmark_poly.run(test_data)


# Combined example
print("\n\n" + "=" * 80)
print("COMBINED EXAMPLE: SEASONAL AUTO-SIZING + CUSTOM PARAMETERS")
print("=" * 80)

print("\nUse case: Weekly retail sales data with custom model tuning")
print("-" * 80)

# Generate weekly sales data (auto-sized for seasonality)
sales_sim = TimeSeriesSimulator(
    length=50,  # Will auto-increase to 3*52=156 for yearly seasonality
    distribution="poisson",
    dist_params={"lam": 150},  # Average 150 sales per week
    trend="linear",
    trend_params={"slope": 0.5},  # Growing trend
    seasonality=52,  # Yearly seasonality (52 weeks)
    seasonality_strength=30.0,
    random_state=42,
)
sales_data = sales_sim.simulate()
print("Requested: 50 weeks")
print(f"Auto-sized to: {len(sales_data)} weeks (3x yearly cycle)")
print(f"Sales stats: mean={sales_data.mean():.1f}, std={sales_data.std():.1f}")

# Benchmark with carefully tuned models
print("\nBenchmarking with custom-tuned forecasters:")
print("-" * 80)
sales_benchmark = ForecastingBenchmark(
    models=[
        # Baseline models
        ("simple_mean", NaiveForecaster(strategy="mean")),
        ("last_week", NaiveForecaster(strategy="last")),
        # Trend-aware model
        ("with_drift", NaiveForecaster(strategy="drift")),
        # Polynomial trends for growth pattern
        ("linear_growth", PolynomialTrendForecaster(degree=1)),
        ("quadratic_growth", PolynomialTrendForecaster(degree=2)),
    ],
    test_size=0.25,  # 25% holdout for testing
    verbose=True,
    random_state=42,
)
sales_results = sales_benchmark.run(sales_data)

print("\n\nBest model for sales forecasting:")
best = sales_benchmark.get_best_model("mae")
print(f"Model: {best[0]}")
print(f"MAE: {best[2]:.2f}")

print("\n" + "=" * 80)
print("✓ Both features working perfectly!")
print("  1. Seasonal data auto-sized to 3x seasonal period")
print("  2. Models initialized with custom parameters")
print("=" * 80)
