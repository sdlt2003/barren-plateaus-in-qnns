"""Centralized default hyperparameters for experiments.

The idea is that experiment scripts keep accepting CLI overrides, but they all
pull their *default* values from a single place so runs stay consistent and are
easy to update.
"""

# Shared defaults
EARLY_STOPPING_TOLERANCE = 1e-3

# Dynamic budgeting
#
# In many papers, a fair compute budget scales with the number of parameters p
# (e.g., O(p)). We implement this as: budget_evals = k * p.
#
# Empirical range suggested: k in [50, 100].
DEFAULT_BUDGET_K = 37.5
MIN_BUDGET_EVALS = 1

# Ideal / noiseless (statevector)
IDEAL_DEFAULT_BUDGET_K = DEFAULT_BUDGET_K
IDEAL_MAX_BUDGET_EVALS = None
IDEAL_EARLY_STOPPING_WINDOW = 100

# Shot-noise / Monte Carlo
SHOT_NOISE_DEFAULT_BUDGET_K = DEFAULT_BUDGET_K
SHOT_NOISE_MAX_BUDGET_EVALS = None
SHOT_NOISE_EARLY_STOPPING_WINDOW = 100
SHOT_NOISE_DEFAULT_TARGET_PRECISION = 0.1

# Real hardware (IBM Runtime)
REAL_HW_DEFAULT_BUDGET_EVALS = 30
REAL_HW_DEFAULT_BUDGET_K = 10
REAL_HW_MAX_BUDGET_EVALS = None
REAL_HW_EARLY_STOPPING_WINDOW = 30
REAL_HW_DEFAULT_SHOTS = 1024
