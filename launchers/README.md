# Stage2 → single-stage bridge launcher

One row of `cases.tsv` = one optimization. Machine-specifics enter only through
`#SBATCH` placeholders (once) and environment variables (per shell) — nothing
in the scripts is hardcoded to a particular machine beyond overridable defaults.

## Quickstart (any SLURM cluster)

```bash
# 1. once: fill the two #SBATCH placeholders
$EDITOR run_stage2_bridge.sbatch        # EDIT_ME_ACCOUNT, EDIT_ME_PARTITION

# 2. pick the runs (one tab-separated row per optimization)
$EDITOR cases.tsv                       # or generate a sweep: ./make_cases.sh

# 3. check, then submit
./submit_bridge_cases.sh --dry-run      # prints exact driver commands, queues nothing
./submit_bridge_cases.sh                # preflights every seed dir, then sbatch array
```

Artifacts land in `OUTPUT_ROOT/G_negative/TF_a_X.XXX_<digest>/` (`metadata.json`
plus the certified Boozer surface). A run that fails any gate publishes
nothing — see `logs/stage2_bridge_*.err`.

## Environment variables (all optional; defaults target Tian's Burg paths)

| variable | meaning |
| --- | --- |
| `PYTHON_BIN` | python from the **simsopt-dipoles fork venv** (`CurrentPenalty`/`ground` are fork-only; upstream simsopt ImportErrors) |
| `DRIVER_PATH` | path to `stage2_to_single_stage.py` |
| `STAGE2_ROOT` | directory containing `TF_a_X.XXX/{bs_opt,surf_opt,results}.json` seed dirs |
| `OUTPUT_ROOT` | where artifacts are published |
| `CASES` | cases file (default: `cases.tsv` next to the scripts) |
| `MAX_ARRAY` / `MAX_CONCURRENT` | queue caps (default 50 / 3) |

## Run without SLURM (workstation)

```bash
PYTHON_BIN=/path/to/fork-venv/bin/python \
DRIVER_PATH=/path/to/stage2_to_single_stage.py \
STAGE2_ROOT=/path/to/seeds OUTPUT_ROOT=/path/to/out \
./submit_bridge_cases.sh --local        # same preflight, rows run sequentially
```

## `cases.tsv` columns

`TF_a  polarity  iota_target  current_limit_A  mpol  ntor  maxiter  step_radius  volume_target`

- `TF_a` selects the seed dir `TF_a_X.XXX/`; `volume_target` of `-` = fitted seed volume.
- `current_limit_A` is a **soft** threshold: currents below it cost nothing.
- Quadrature/certification densities derive from `mpol`/`ntor` automatically.
- Runtime measured at `mpol=ntor=6`: ~3 min/case on a 32-core workstation
  (`TF_a = 0.4`). Other orders and cluster walltimes are unmeasured — the 2 h
  SLURM limit is a deliberate ~40× safety factor.
