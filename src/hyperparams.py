"""Centralized default hyperparameters for experiments.

The idea is that experiment scripts keep accepting CLI overrides, but they all
pull their *default* values from a single place so runs stay consistent and are
easy to update.
"""

# Shared defaults
EARLY_STOPPING_TOLERANCE = 1e-3
EARLY_STOPPING_WINDOW = 200

# Dynamic budgeting
#
# In many papers, a fair compute budget scales with the number of parameters p
# (e.g., O(p)). We implement this as: budget_evals = k * p.
#
# Empirical range suggested: k in [50, 100].
DEFAULT_BUDGET_K = 37.5

MIN_BUDGET_EVALS = 1

# Shared defaults for simulation scenarios (ideal + shot-noise)
SIM_DEFAULT_BUDGET_K = DEFAULT_BUDGET_K
SIM_EARLY_STOPPING_WINDOW = EARLY_STOPPING_WINDOW
SIM_MAX_BUDGET_EVALS = None

# Shot-noise / Monte Carlo specific
SHOT_NOISE_DEFAULT_TARGET_PRECISION = 0.1

# Real hardware (IBM Runtime)
# Dynamic default: budget_evals = round(k * num_params), same rule as simulation.
REAL_HW_DEFAULT_BUDGET_K = 20
# Legacy fixed profile (only if budget mode is switched to "fixed" in exp-real.py).
REAL_HW_FIXED_BUDGET_BASE = 32
REAL_HW_MAX_BUDGET_EVALS = None
REAL_HW_EARLY_STOPPING_WINDOW = 30
REAL_HW_DEFAULT_SHOTS = 1024
# IBM docs: wait for job.result() without a client-side timeout unless debugging.
REAL_HW_RUNTIME_RESULT_TIMEOUT = None
# Default IBM Runtime container for real-hw grids: one Batch per (seed, q) Slurm task.
REAL_HW_EXECUTION_MODE = "batch"  # session | batch | job
# None = do not pass max_time to Session()/Batch(); IBM uses the plan default (often ~8h Premium).
REAL_HW_RUNTIME_MAX_TIME = None
# Backward-compatible alias used by older CLI/env names.
REAL_HW_SESSION_MAX_TIME = REAL_HW_RUNTIME_MAX_TIME
