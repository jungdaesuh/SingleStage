"""
Epsilon-constraint parameter generator for single-stage Pareto front identification
using a continuation schedule over current weights.

Instead of randomly sampling all scalarization weights and identifying the Pareto
front post-hoc (which wastes most evaluations on dominated solutions), this script
systematically sweeps iota_target as a near-hard constraint and minimizes max dipole
current at each target value, directly tracing the Pareto front.

The idea:
  - The Pareto front is a 1D curve in (iota, max_current) space.
  - By fixing iota_target and using a high iota_weight, each evaluation finds the
    minimum achievable current at that iota — i.e., a point ON the Pareto front.
  - Sweeping iota_target across the range of interest traces the entire front.
  - For each iota_target, run a continuation (warm-start) sweep over a small
    schedule of current weights inside single_stage_dipole_example.py. This
    yields multiple solutions with varying Boozer residual / current tradeoff,
    and you can choose a Boozer-residual cutoff a posteriori.

Typical usage with batch_scan.sh:
    N=$(python3 generate_inputs.py --total)
    PARAMS=$(python3 generate_inputs.py --index $i)

To inspect all parameter combinations:
    python3 generate_inputs.py --list
"""

import numpy as np
import argparse
import sys


# ============================================================================
# Epsilon-Constraint Sweep Configuration
# ============================================================================
# Iota targets: evenly spaced across the range of interest.
IOTA_TARGETS = np.linspace(0.10, 0.25, 16)

# Continuation schedule of current weights (passed through to
# single_stage_dipole_example.py as --current-weight-schedule).
CURRENT_WEIGHT_SCHEDULE = [0.25, 0.5, 1.0, 2.0]

# Fixed parameters (not swept):
# High iota weight enforces the iota target as a near-hard constraint so that the
# optimizer lands close to the prescribed iota value.
IOTA_WEIGHT = 1.0

# Moderate QS weight: slightly less than iota, but still enough to maintain
# good quasi-symmetry to within the 5% increase threshold for the QS objective.
QS_WEIGHT = 1.0

def total_sweep_points():
    """Return the total number of sweep points (iota targets)."""
    return len(IOTA_TARGETS)


def generate_params(index):
    """
    Generate parameters for sweep point `index`.

    Each index selects an iota target; current-weight continuation is handled
    inside the optimization script.
    """
    n_total = total_sweep_points()

    if index < 0 or index >= n_total:
        print(f"Error: index {index} out of range [0, {n_total})", file=sys.stderr)
        sys.exit(1)

    return {
        "IOTA_TARGET": float(IOTA_TARGETS[index]),
        "IOTA_WEIGHT": IOTA_WEIGHT,
        "QS_WEIGHT": QS_WEIGHT,
        "CURRENT_WEIGHT_SCHEDULE": ",".join(str(x) for x in CURRENT_WEIGHT_SCHEDULE),
    }


# ============================================================================
# Original random sampling (kept for reference / backward compatibility)
# ============================================================================
def random_number(low, up):
    """
    Draw a random number from a uniform distribution between low and up
    """
    return np.random.rand() * (up - low) + low

def random_number_exp(low, up):
    """
    Draw a random number from a log-uniform distribution between low and up
    """
    return 10 ** random_number(low, up)

def generate_params_random() -> dict:
    """
    Generate a random set of targets and weights for a single-stage run.
    These have to be fine tuned for each initial condition

    """
    # TARGETS
    current_threshold = random_number(125000, 200000)
    iota_target = random_number(0.08, 0.14)

    # WEIGHTS
    # Iota penalty weight: ~50–500
    iota_weight = random_number_exp(1.7, 2.7)

    # QS (nonQS ratio) weight: ~1–300
    qs_weight = random_number_exp(0.0, 2.5)

    # Current penalty weight: ~0.01–1 (gradient is now active via vjp fix;
    # per-coil gradient in DOF space is O(0.1), so weight >1 dominates)
    current_weight = random_number_exp(-2.0, 0.0)

    return {
        "CURRENT_THRESHOLD": float(current_threshold),
        "IOTA_TARGET": float(iota_target),
        "IOTA_WEIGHT": float(iota_weight),
        "QS_WEIGHT": float(qs_weight),
        "CURRENT_WEIGHT": float(current_weight),
    }


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parameter generator for single-stage Pareto front sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_inputs.py --list                   # show all sweep points
  python3 generate_inputs.py --total                  # print total count (for batch_scan.sh)
  python3 generate_inputs.py --index 5                # CLI args for sweep point 5
  python3 generate_inputs.py --mode random             # single random sample (original behavior)
""",
    )
    parser.add_argument(
        "--mode",
        choices=["epsilon-constraint", "random"],
        default="epsilon-constraint",
        help="Sampling strategy (default: epsilon-constraint).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Sweep point index [0, N) for epsilon-constraint mode.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all epsilon-constraint parameter combinations and exit.",
    )
    parser.add_argument(
        "--total",
        action="store_true",
        help="Print total number of sweep points and exit (useful for batch_scan.sh).",
    )
    args = parser.parse_args()

    # --total: just print the count so batch_scan.sh can use it as N
    if args.total:
        print(total_sweep_points())
        sys.exit(0)

    # --list: show the full sweep grid
    if args.list:
        n = total_sweep_points()
        print(f"Total sweep points: {n}")
        print(f"  {len(IOTA_TARGETS)} iota targets\n")
        hdr = f"{'idx':>4}  {'iota_target':>12}  {'iota_weight':>12}  {'qs_weight':>10}  {'cw_schedule':>18}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(n):
            p = generate_params(i)
            print(
                f"{i:4d}  {p['IOTA_TARGET']:12.4f}  {p['IOTA_WEIGHT']:12.1f}"
                f"  {p['QS_WEIGHT']:10.1f}  {p['CURRENT_WEIGHT_SCHEDULE']:>18s}"
            )
        sys.exit(0)

    # Generate parameters based on mode
    if args.mode == "random":
        params = generate_params_random()
    else:
        if args.index is None:
            print(
                "Error: --index is required for epsilon-constraint mode "
                "(or use --list / --total)",
                file=sys.stderr,
            )
            sys.exit(1)
        params = generate_params(args.index)

    # Print CLI args for single_stage_dipole_example.py (captured by batch_scan.sh)
    cli_args = (
        f"--iota-target {params['IOTA_TARGET']} "
        f"--iota-weight {params['IOTA_WEIGHT']} "
        f"--qs-weight {params['QS_WEIGHT']} "
        f"--current-weight-schedule {params['CURRENT_WEIGHT_SCHEDULE']}"
    )
    print(cli_args)
