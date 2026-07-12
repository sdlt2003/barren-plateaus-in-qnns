"""Centralized default hyperparameters for experiments.

The idea is that experiment scripts keep accepting CLI overrides, but they all
pull their *default* values from a single place so runs stay consistent and are
easy to update.
"""

# Dynamic budgeting
#
# In many papers, a fair compute budget scales with the number of parameters p
# (e.g., O(p)). We implement this as: budget_evals = k * p.
#
# Empirical range suggested: k in [50, 100].
DEFAULT_BUDGET_K = 37.5

# Shared defaults for simulation scenarios (ideal + shot-noise)
SIM_DEFAULT_BUDGET_K = DEFAULT_BUDGET_K
SIM_MAX_BUDGET_EVALS = None

# Shot-noise / Monte Carlo specific
SHOT_NOISE_DEFAULT_TARGET_PRECISION = 0.1

# Inline gradient checkpoints during simulation training (--track-grad-* in exp-ideal/shot-noise).
SIM_GRAD_CHECKPOINT_STRIDE = 10

# Real hardware (IBM Runtime)
REAL_HW_DEFAULT_BUDGET_K = 20
REAL_HW_MAX_BUDGET_EVALS = None
REAL_HW_DEFAULT_SHOTS = 1024
REAL_HW_MAX_SHOTS = None

# None = wait for job.result() without a client-side timeout (IBM-recommended on HW).
REAL_HW_RUNTIME_RESULT_TIMEOUT = None

# IBM Runtime container for Estimator(mode=...). Options:
#   batch   - jobs grouped in a Batch; good queue priority for many independent
#             grid points (default for Slurm: one Batch per seed x qubits task).
#   session - reserved QPU access for the container TTL; useful when chaining
#             many estimator calls on the same backend in one process.
#   job     - no container; each estimator.run is a standalone job (simplest,
#             but typically slower queueing for large grids).
REAL_HW_EXECUTION_MODE = "batch"  # batch | session | job

# Optional TTL for session/batch (e.g. "8h"). None = IBM plan default.
REAL_HW_RUNTIME_MAX_TIME = None
# Backward-compatible alias for older CLI/env names (--session-max-time).
REAL_HW_SESSION_MAX_TIME = REAL_HW_RUNTIME_MAX_TIME
