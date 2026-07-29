# Stage2 → Single-Stage Bridge: Fix Plan (job 9209252 post-mortem)

**Status:** In progress — engineering complete and validated locally 2026-07-29: Phases 0/2/3 implemented (`d7f925a`, `f04e14c`), three further defects found and fixed by the measurements themselves (gradient-gate norm `d4a152b`, solve-grid under-resolution `88879d1`, current-penalty semantics `5e2c27c`), and the fixed driver **publishes a fixed-order validated artifact end-to-end on the local seed** (iota error 2.0e-4, zero rejected trials — ablation E5i). The E5 receipts remain local-only and are not shipped by `5982139`. Remaining: Phase 1 (delivery — nothing is pushed) and Tian's policy confirmations (150 kA soft-threshold value, 11 poloidal rows, TF-DOF scope).
**Last updated:** 2026-07-29

> **Interpreter (read before running anything).** Use
> `/home/jungdaesuh/code/columbia/dipole/tian/simsopt-dipoles/.venv/bin/python`.
> `CurrentPenalty` is **fork-only** — it does not exist in upstream simsopt, so every
> driver in this repo raises `ImportError` under `opensource/simsopt`. The pinned build
> is `simsopt 0.1.dev1+g9c2771cdb` with `ground` 9.0.0 (required by
> `is_self_intersecting`), per `environment_receipt.json`.

## Purpose

Job 9209252 ran 11 h 29 m 48 s and produced a scientifically void result. This plan
records what is fixed on local `main`, what is genuinely still open, and how to prove
the fixes work — without repeating the unvalidated speedup estimates that the first two
analysis passes made.

**The fixes are not delivered.** `origin/main` is still `49d47d7`; every fix below lives
in unpushed local commits, and the one open PR does not touch this driver. See
Current Context.

The archived driver is `stage2_to_single_stage_tian_9209252.py`. **Cite `d610cdc` for it,
not the worktree** — `d610cdc` holds the 297-line file Tian ran; `d79cc76`
reordered the same path into 337 lines. The maintained driver is
`stage2_to_single_stage.py`.

## Goals

- Every open defect from the 9209252 post-mortem is either fixed or explicitly deferred with a reason.
- **The objective is dimensionally commensurable**: every term is divided by a declared SSOT scale before weighting, so iota and quasi-symmetry can actually influence the optimum. Unit weights alone do not achieve this.
- `stage2_to_single_stage.py` bounds its outer step **in normalized coordinates**, so line-search trials cannot leave the valid Boozer branch and the bound has physical meaning.
- One SSOT for the rejected-trial model, not two divergent copies.
- A measured before/after runtime comparison from `PhaseJournal`, replacing estimates.
- The stage2-bridge fixes reach `origin` in a reviewable form. They are currently unpushed local commits and are in no PR.

## Non-Goals

- Repairing `stage2_to_single_stage_tian_9209252.py`'s *logic*. It is forensic evidence.
  **But it is not currently frozen, and that needs deciding (see Open Questions):**
  `d610cdc` added the 297-line file; `d79cc76` then reordered it into 337
  lines (+241/−201). The worktree copy no longer matches what Tian ran.
  **`d610cdc` is the closest faithful forensic source** — 8,758 bytes / 297 lines against
  the original download's 8,759 bytes / 298 lines. The sole difference is one trailing
  blank line; all 297 content lines are identical. It is *not* byte-exact, so do not
  describe it that way.
- Re-running job 9209252 to reproduce the failure. The defect is proven statically.
- Changing the physics targets (iota, current p-norm, volume) beyond correcting the plumbing that misapplies them.
- Order-12 / resolution-ladder work. Tracked separately in `stage2_single_stage_24core_implementation_plan.md`.

## Current Context

Repo state at time of writing (`git rev-list`, `gh pr view`, 2026-07-28):

| ref | commit | note |
| --- | --- | --- |
| `origin/main` (tianlangg) | **`49d47d7`** | **Tian's repo contains only the Initial commit** |
| local `main` | moves | **Many commits ahead of `origin/main`, none pushed.** Check with `git rev-list --count origin/main..main` — this plan's own commits change it, so no number is pinned here. |
| `tian-minimal` | `4fee61e` | 8 commits, branched from `49d47d7`, pushed to `fork` |
| merge-base(`main`,`tian-minimal`) | `49d47d7` | the two lines never shared work |

**Read this before trusting the table below.** Nothing in it has been delivered to Tian.
The fixes exist only as unpushed local commits on a branch nobody has reviewed:

- `stage2_to_single_stage.py` **is present on `tian-minimal`, but it is Tian's unfixed original** — 282 lines, byte-identical to `origin/main`'s copy. `tian-minimal` therefore carries the bug, not the fix. (Corrected 2026-07-28: an earlier draft said the file was absent. That came from a broken shell check — `grep -c … || echo ABSENT` fires the `||` branch because `grep -c` exits 1 on a zero count. Verify presence with `git cat-file -e <ref>:<path>`, never with `grep`'s exit status.)
- PR #1 (`jungdaesuh:tian-minimal` → `tianlangg:main`, OPEN, MERGEABLE, +1097/−254, 9 files) touches `boozer_functions.py`, `bounded_bfgs.py`, `run_configuration.py`, `run_optimize_{scan,single}.py`, `single_stage_dipoles{.py,.sh}`, `tests/test_bounded_bfgs.py`, `tests/test_minimal_fixes.py`. **It carries none of the stage2-bridge work.**
- PR #1's MERGEABLE status says nothing about the local commits. GitHub computes it against `origin/main` (`49d47d7`) and has no knowledge of anything unpushed. It is not evidence that the two lines are reconciled.
- The file sets are **not** disjoint either: PR #1 and `origin/main..main` both touch `boozer_functions.py`, `single_stage_dipoles.py`, and `single_stage_dipoles.sh`. Any merge order has to account for those three.

Fixed on **local `main`** in `stage2_to_single_stage.py` (verified by reading the file at
`d79cc76`; *not* on `origin/main`, *not* in PR #1):

| Post-mortem defect | Where it is fixed |
| --- | --- |
| Stage-2 surface fit discarded | `stage2_contracts.build_boozer_solve_surface` — fit lands on the returned `surface` |
| `targetlabel` = default-torus volume | same; follows from the fit being applied |
| ~~Weights `1e-2 / 100 / 100` starve iota~~ | **NOT FIXED — see Phase 0.** Unit weights do not fix this. |
| Incoherent `dJ = -prev_dJ` | `stage2_contracts.rejected_trial_value_and_gradient` — coherent quadratic model |
| Full BFGS+Newton per outer trial | `stage2_contracts.solve_boozer_outer_trial` — *"Warm-start one outer-optimizer trial using Newton only"* |
| 64×64 grid (4,096 points) | `build_boozer_solve_surface` — `(6·ntor+1) × (6·mpol+1)` since `88879d1`. The earlier `(4·ntor+1) × (4·mpol+1)` replacement was later measured to be under-resolved. **The half-period *domain* was never a defect** — simsopt supports it first-class as `Surface.RANGE_HALF_PERIOD`. The archive's grid is merely the *unshifted* variant: simsopt's half-period range adds `0.5·dphi` "in order to provide spectral convergence of integrals" (`simsopt/geo/surface.py:168-171`), which `np.linspace(0, 1/(2·nfp), 64, endpoint=False)` omits. |
| Publishes on failure | `stage2_to_single_stage.py:293-303` + `certify_boozer_solution` |
| Hardcoded `sign_g`, `abs()` currents | `stage2_contracts.apply_field_polarity` + `--field-polarity` |
| Orphan `resolution = 6` | `RESOLUTION` constant, actually used |
| No per-solve timing | `PhaseJournal` + `BoozerPhaseReport.elapsed_seconds` → JSONL, aggregated as `boozer_total_seconds` |

### Local Stage-2 seed (verified present 2026-07-28)

The full three-file contract the bridge consumes is on this machine:

```
/home/jungdaesuh/code/columbia/dipole/tian/outputs/stage2_initial_conditions/
  wout_nfp22ginsburg_000_001242/02_ntf4_diprad_0.05_VVa_0.26_VVb_0.27_VV_R0_1.03/
    bs_opt.json   surf_opt.json   results.json
```

| quantity | value |
| --- | --- |
| equilibrium | `wout_nfp22ginsburg_000_001242`, `surf_s = 0.9` — **the one Tian used** |
| **`TF_a`** | **0.4 — Tian ran 0.600** |
| coils | 368 = 16 TF (`ntf=4`, `nfp=2`) + 352 dipoles (`num_wps`) |
| TF current | 158,827.07 A, all identical (`field_on_axis = 0.5`) |
| dipole currents | 179.10 … 110,990.70 A (`min/max_wp_current`) |
| `surf_volume` | 0.32670472159209635 m³ |
| Stage-2 status | `success: True`, converged, `final_squared_flux` 6.373e-05 |

Two independent cross-checks that this seed is the right physical object:

1. `surf_volume` = 0.3267047 matches the volume computed from the wout at s=0.9 to seven
   digits — the same number that makes the archive's 0.19739 m³ target 60% of correct.
2. **G reproduces the log.** 16 × 158,827.07 A × μ₀ = **3.193408**, against the logged
   `G = −3.193422` in job 9209252 — agreeing to 1.4e-05 despite the different `TF_a`,
   because `field_on_axis` fixes the TF current.

**Scope limit — state this whenever quoting a number against the 9209252 log.** `TF_a`
differs (0.4 vs 0.600), so runs on this seed are **not** a numerical reproduction of that
job. They are valid *ablations*: every variant runs against the same baseline, which is
what an ablation requires. Do not compare an absolute value from this seed to the log.

### Measured facts from the 9209252 log

- 11 h 29 m 48 s wall clock; 42 nested Boozer solves; 16,268 inner BFGS iterations.
- 19 of 42 trials rejected (45%), **all for self-intersection, zero for residual failure**.
- Rejected trials: 10,998 inner iterations (**67.6%**), mean 579 vs 229 for valid ones.
- 3 outer optimizer steps out of `maxiter=150`; terminated `Desired error not necessarily achieved due to precision loss`.
- Achieved `||grad||_inf` after Newton: median 1.82e-15, 34/42 at ≤1e-14.
- Newton iteration counts across the 42 solves: `{0: 1, 1: 37, 2: 4}` — one no-op, the rest 1–2.
- Grid: the archive used 64×64 = **4,096** points at `mpol=4, ntor=5`. The current 6×
  policy at those same orders would be 31×25 = 775 (**5.29×** fewer points); the
  maintained driver runs `RESOLUTION = 6`, so it builds 37×37 = **1,369** points
  (**2.99×** fewer than the archive). The earlier 21×17 and 25×25 figures describe the
  retired 4× policy, not the maintained driver.
- `targetlabel` = 0.19739 m³ ⇒ **s ≈ 0.56** (bracketed by s=0.55 → 0.194863 and
  s=0.60 → 0.213271; linear interpolation gives 0.557), against the intended s = 0.90
  cut (0.32670 m³).
- iota term at first callback = 3.9e-4 of J = 9.0e6 → weight ratio **4.4e-11**.

**The objective was ~100% the current penalty.** `CurrentPenalty.J()` is a p-norm *in
amperes*, `(Σ|Iᵢ|^p)^(1/p)`, which at `p=12` approaches `maxᵢ|Iᵢ|`. With
`CURRS_WEIGHT = 100`, dividing the three logged callback values by 100 recovers a
physically coherent, monotonically shrinking dipole current:

| callback | J | J / 100 |
| --- | --- | --- |
| 1 | 9,011,084 | 90,110.8 A |
| 2 | 7,961,391 | 79,613.9 A |
| 3 | 7,496,082 | 74,960.8 A |

**INFERENCE, not measurement.** The log prints only the total `J`, so the split is
argued, not observed:

- `CurrentPenalty` returns amperes; dipole currents here are O(10⁴–10⁵ A) (cf. the
  `fcp150kA` target in `stage2_single_stage_24core_implementation_plan.md:79`), so
  `CURRS_WEIGHT × J_current` is O(10⁶–10⁷) — exactly the observed magnitude.
- `NonQuasiSymmetricRatio` is a ratio, O(1), at weight 1.0.
- The iota penalty is 3.9e-4, computed exactly from the logged iota.
- **`BoozerResidual` is the gap:** its magnitude cannot be recovered from the log. At
  `BZRES_WEIGHT = 100` it would need to reach O(10⁴) to rival the current term — a very
  large residual for a solve that converged to `||grad||_inf ~ 1e-15` — but this is
  unverified.

**MEASURED 2026-07-29 (ablation E1, local `TF_a = 0.4` seed):** the argument above is
confirmed. At the initial Boozer solution the raw terms are current **126,048.57**
(p=12 norm in amperes), iota penalty 0.037146, nonQS 4.02e-05, `BoozerResidual`
**1.62e-05** — the residual's share is ~1.3e-10 under both the archive's weights and
unit weights, and the current term's share is ≥ 0.9999997 under both. The gap is
closed; the dominance claim is no longer an inference. Full table:
`stage2_to_single_stage_ablation_studies_implementation_plan.md`, Results E1.

## Rationale

The correctness defects are fixed **in local commits that no one else can see**. That is
the first problem, and it outranks the rest: unpushed work is indistinguishable from no
work, and Tian is still running the pre-fix script. Phase 1 is therefore delivery, not
engineering.

What remains after that is a genuine code gap (outer step bounding), an SSOT violation,
two unresolved design questions, and the absence of any *measured* evidence that the
fixes deliver. The first two analysis passes both produced runtime estimates ("~1 hour",
"15–20 minutes") that were extrapolations from point count, not results. This plan
refuses to close on estimates.

Outer step bounding is ordered after correctness because it is a performance and
robustness measure, not a correctness one. It is one of **two** remaining code defects —
the other is the unnormalized objective (Phase 0), which is a correctness defect and
outranks it.
It matters because the surface degenerates *during* the nested solve, not before it:
there is nothing to screen ahead of time, so the only real remedies are bounding the
trial step, guarding inside the solve loop, or capping doomed solves cheaply.

## Assumptions

- **ASSUMPTION:** per-inner-iteration cost is roughly uniform. The 67.6%-of-iterations
  figure is measured; any conversion to hours is *derived* and must not be reported as
  measured. Newton steps (1–2 per solve, Hessian assembly + LU) are excluded from the
  iteration count, and `JF.J()/dJ()` runs only on valid trials — both bias a naive
  hours split toward 50/50.
- **HYPOTHESIS — partially settled 2026-07-29 (ablations E1/E2, `TF_a = 0.4` seed).**
  Measured now: the per-term split (current dominates, ≥ 0.9999997 under both weight
  sets — E1), and the counterfactual weight A/B (E2): iota collapses identically under
  archive and unit weights (0.0274 → 0.015/0.014 with ~40 % rejection and the same
  precision-loss exit as 9209252), so the collapse is caused by the unnormalized current
  term, **not** by the archive's particular weight choice — E2's pre-registered
  differential criterion is formally falsified. Still unproven: the link from iota
  magnitude to *self-intersection* — every local rejection is Newton non-convergence,
  none self-intersection, so that final arrow of the causal chain remains
  `TF_a = 0.600`-only and unmeasured. Whether Phase-0 normalization suppresses the
  collapse is E5, blocked on Phase 0 existing.
- **FACT:** Stage-2 inputs (`bs_opt.json`, `surf_opt.json`, `results.json`) live under
  `/burg-archive/...`. **A complete local seed also exists — see "Local Stage-2 seed"
  below. End-to-end validation is NOT cluster-only.** (Corrected 2026-07-28: an earlier
  draft claimed no local Stage-2 data. The search was run inside the checkout; the seed
  lives one directory *above* it, exactly as `run_local_stage2_case.py`'s docstring
  states — "keeping the output outside the checkout".)
- **POLICY:** production iota target is 0.05, from the path
  `.../iota0.05_fcp150kA_vt0.3/...` in `stage2_single_stage_24core_implementation_plan.md:79`.
  E5c measured that 0.3 is wrong for this seed; E5i validated 0.05. Direct driver calls
  now require the target explicitly, and the Ginsburg launcher passes 0.05.

## Implementation Plan

0. **Normalize the objective — the defect is still live** *(critical; verified 2026-07-28)*

   `stage2_to_single_stage.py:196-199` sets every weight to 1.0 and applies **no scale
   normalization**. That does not fix iota starvation, because the terms are not
   commensurable: `CurrentPenalty` returns **amperes**, `NonQuasiSymmetricRatio` and the
   iota `QuadraticPenalty` are **dimensionless**. At unit weights, using the logged run:

   | term | weight 1.0 value | units |
   | --- | --- | --- |
   | current | **90,110.84** | A |
   | nonQS | ~1 | — |
   | iota | **0.039453** | — |

   Current/iota = **2.28e6**, ~6 orders of magnitude. Changing `1e-2 / 100 / 100` → `1.0`
   moved the ratio from 4.4e-11 to 4.4e-7 of the objective; iota is still absent.

   **DONE 2026-07-29, commit `d7f925a`.** `I_scale` fixed at **150,000 A** by user
   decision (production `fcp150kA`); iota curvature is `iota_penalty_curvature(target)`
   = 1/|target|², both in `stage2_contracts.py` (SSOT), units stated at the `JF`
   composition, formulas pinned by `tests/test_objective_scales.py`. Measured composed
   terms at the seed's initial solution: current 0.84, iota 0.41, nonQS 4e-5,
   residual 2e-5 — commensurable. Flag the 150 kA value to Tian on delivery.
   - [x] Explicit SSOT scales; terms dimensionless before weighting.
   - [x] Precedent reused (`FCP_THRESHOLD` family / `IOTA_WEIGHT / IOTA_SCALE**2` form); value confirmed by the user, still to be confirmed by Tian.
   - [x] Scales in `stage2_contracts`, not inline literals.
   - [x] Units stated in the `JF` comment.
   - [x] Landed as code, with tests.

1. **Get the stage2-bridge work in front of Tian at all**
   - [ ] Decide what ships. Today PR #1 carries the `single_stage_dipoles.py` line only; every fix in the table above is unpushed and unreviewed. Nothing about this bridge has reached `origin`.
   - [ ] **Do not push local `main` wholesale.** Check the size first — `git rev-list --count origin/main..main` and `git diff --shortstat origin/main..main`. At the last check it was two dozen files and roughly +7.8k lines, dragging in unrelated SQP/KKT and single-stage machinery. That is not a reviewable PR. *(Deliberately not pinning exact numbers: this plan's own commits change them.)*
   - [ ] Instead cut a clean branch from `origin/main` and cherry-pick only the Stage-2 bridge surface. Minimum file list: `stage2_to_single_stage.py`, `stage2_contracts.py`, `coil_layout.py`, `spectral_policy.py` if referenced, the launcher, the focused tests — **and `bounded_bfgs.py`, which Phase 3 requires.** Record the final list here before pushing.
   - [ ] **Resolve `bounded_bfgs.py` ownership before cutting the branch.** It exists only on `tian-minimal` and is already in PR #1; `origin/main` and local `main` both lack it (`git cat-file -e <ref>:bounded_bfgs.py`). So the Stage-2 PR must either land *after* PR #1 merges, or it will introduce a second copy — recreating the exact SSOT violation Phase 2 exists to remove. **Sequencing after PR #1 is the default.**
   - [ ] Verify the curated branch's diff is confined to that list (`git diff --stat origin/main..<branch>`) before opening the PR.
   - [ ] Confirm PR #1 remains mergeable **against the current remote base** before sequencing the Stage-2 PR. Its status reflects `origin/main` only; local commits are invisible to it and are not what makes it mergeable.
   - [ ] Do **not** rebase `tian-minimal` onto local `main` without confirming with the user: `4fee61e` is already pushed to `fork` and is the open PR head.

2. **Close the SSOT violation on the rejection model**

   **DONE 2026-07-29, commit `f04e14c`.** `bounded_bfgs.py` (with its tests) imported
   verbatim from `tian-minimal`, so the file is byte-identical to PR #1's copy and
   cannot conflict on merge.
   - [x] The two copies were diffed first — semantically identical, no drift.
   - [x] One definition, owned by `bounded_bfgs.py`; the `stage2_contracts` copy is deleted.
   - [x] All call sites updated: the driver, `single_stage_dipoles.py`, `tests/test_stage2_contracts.py`.

3. **Bound the outer step — but normalize the coordinates first** *(critical prerequisite)*

   `minimize_bounded_bfgs` bounds a plain Euclidean step: `step_length = min(1.0,
   step_bound / np.linalg.norm(direction))` (`bounded_bfgs.py:70-71`). For
   `single_stage_dipoles.py` that is sound because its outer DOFs are homogeneous. **It is
   not sound here.** This driver frees dipole *currents* (amperes,
   `stage2_to_single_stage.py:129`) **and** TF *curve* coordinates (metres, `:132`), so an
   L2 radius over the raw vector adds A² to m² and has no physical meaning — the bound
   would be dominated by whichever DOF family happens to carry larger numbers.

   **DONE 2026-07-29, commit `f04e14c` (route (b)), gate corrected in `d4a152b`.**
   - [x] Decision: **(b)** — TF curves fixed, outer vector = dipole `ScaledCurrent` multipliers only. The seed stores each dipole current as `ScaledCurrent(Current(1.0), I_stage2)`, so the vector is dimensionless fractions and the L2 radius is meaningful without extra scaling. Measured justification (ablation E5a/E5b): with TF curves free the normalized outer optimization accepts **zero** steps; currents-only runs 12/12 with monotone descent.
   - [x] `minimize_bounded_bfgs` wired in, replacing the bare scipy BFGS.
   - [x] `--outer-step-radius` CLI flag, default 0.15, units documented (dimensionless multipliers), validated finite-positive.
   - [x] The every-evaluated-trial-within-radius test ships in `tests/test_bounded_bfgs.py` (imported with the module).
   - [x] **Latent gate bug found by the first converged run and fixed (`d4a152b`):** the outer gradient gate measured the L2 norm against `gtol`, but the optimizer's success contract is the **inf norm** — stricter by up to √dim; it rejected a legitimately converged 176-DOF run. The gate now re-verifies the contract the optimizer promises, and a test pins that contract.

4. **Instrument the trial path** *(telemetry only — this phase does not itself make failures cheaper; the mechanism is Phase 3's step bound plus the existing `NEWTON_MAXITER` cap)*
   - [ ] **Verified 2026-07-28:** `NEWTON_MAXITER = 40` (`stage2_contracts.py:27`) *is* applied on the trial path — `solve_boozer_outer_trial` → `solve_boozer_newton` (`:457`) → `maxiter=NEWTON_MAXITER` (`:473`). Remaining question is whether 40 is the right cap, not whether a cap exists.
   - [ ] Record rejected-trial elapsed time separately in `PhaseJournal` so the wasted fraction becomes **measured** rather than inferred.
   - [ ] Log the per-term objective breakdown (nonQS / iota / current / residual) per callback, so the composition argument above becomes a measurement.
   - [ ] Do **not** add a pre-solve self-intersection screen. It cannot work: the surface is valid before the solve and degenerates during it.

5. **Resolve the blocking design questions** (Open Questions lists six; these two are the ones that gate code in this plan — the archive freeze and `I_scale` gate Non-Goals and Phase 0 respectively, and are tracked there)
   - [ ] TF coil DOF scope: `stage2_to_single_stage.py:132` frees *every* TF curve DOF (`coil.curve.unfix_all()`). Identical on `main` and in the archive, so it is inherited from the Wataru template, not a Tian regression. Decide whether the intent is full coil shape freedom or only radial translation/rotation.
   - [x] `--iota-target` has no hidden driver default; direct calls must provide it, and the Ginsburg launcher explicitly passes the production value 0.05.

6. **Measure, then claim** *(runnable locally — see "Local Stage-2 seed")*

   The experiment design lives in
   [`stage2_to_single_stage_ablation_studies_implementation_plan.md`](stage2_to_single_stage_ablation_studies_implementation_plan.md).
   That plan owns the six ablations; this phase owns folding their results back here.

   - [x] **Prerequisite defect — `coil_layout.py:14` — RESOLVED 2026-07-29.** The user
     ruled the local seed canonical; `dipole_poloidal_rows` is now `11` (commit
     `b367f36`), the gate passes on the seed (352 = 32 × 11), tests updated and green
     (85). The gate itself and the toroidal invariant were not touched. Note this chose
     "correct the SSOT constant" over the previously preferred "derive from
     `results['npoloidal']`" — the SSOT stays a policy declaration, not an echo of its
     input. If Tian rules the seed non-canonical, revert `b367f36`.
   - [x] Baseline established on the local seed: 4 outer steps / 64 trials / 42.2 %
     rejected / 1,970 inner iterations / 3,383 s wall, precision-loss exit (ablation
     plan, Results E2). Caveat: the current driver *configuration*, not the archive
     driver — and run through the instrumented harness transcription disclosed there.
   - [x] **Harness validated:** `G` identity reproduced to 1.3e-10 (ablation Results E0).
   - [x] Ablate one change at a time against that baseline — **measured 2026-07-29** (ablation plan, E5 ladder): normalization alone cures the collapse but accepts zero steps; + currents-only restores descent; the full fixed driver converges at the production iota target (148 iterations, 15 % rejection vs 42 % baseline). Certification then exposed a further live defect — see the solve-grid item below.
   - [x] Per-phase `elapsed_seconds` extracted from each run's `phase_records_*.jsonl` (consolidated in `ablation_runs/consolidated_metrics.json`).
   - [x] Wall clock, solve counts, inner iterations, rejected fractions reported as same-seed deltas — ablation plan, Results E2/E4. Headlines: legacy rejection model dies at 3 outer steps (9209252's count) vs 4; full BFGS+Newton per trial costs 16.6× the inner work of Newton-only; the 4,096-point grid costs only 1.16× median per-solve wall on the Newton-only path.
   - [ ] Only if a cluster run is later needed for the exact `TF_a = 0.600` case, repeat there. **Sharpened by E2's scope note: the self-intersection rejection pathway did not occur at `TF_a = 0.4` at all, so it can only be studied at 0.600.**

## Validation Plan

Local (no cluster data required). **All of these require the `simsopt-dipoles` venv named
in the header — they fail with `ImportError: CurrentPenalty` under any other interpreter:**

- [ ] `/home/jungdaesuh/code/columbia/dipole/tian/simsopt-dipoles/.venv/bin/python -m pytest tests/ -q` passes on the reconciled branch.
- [x] Unit test: **the normalization formulas**, not the term shares. A "no term exceeds 99% of `JF.J()`" check is invalid — at the iota target the iota penalty is exactly zero, so another term legitimately contributes 100%. The tests instead pin the reference deviations: an iota error of one `IOTA_SCALE` contributes 0.5; current at the 150 kA limit contributes zero; current one full scale above the limit contributes 0.5.
- [ ] Unit test: `build_boozer_solve_surface` output has >3 nonzero DOFs and is *not* equal to a fresh default `SurfaceXYZTensorFourier` — this is the exact regression that caused 9209252.
- [ ] Unit test: for a source surface, `abs(built.volume())` matches `abs(source.volume())` to a stated tolerance (guards `targetlabel`).
- [ ] Unit test: `rejected_trial_value_and_gradient` returns a value strictly greater than the accepted value for any nonzero displacement, and a gradient that is the true gradient of the returned model.
- [ ] Unit test: outer step bound is respected.
- [ ] Grep gate: `sign_g` and a bare `abs(coil.current.get_value())` polarity computation appear nowhere in the maintained driver.

End-to-end, **on the local seed** — point `--stage2-root` at the directory in "Local
Stage-2 seed". No cluster access required:

- [x] Fixed driver completes without a publication-gate failure (2026-07-29, local-only ablation E5i: fixed-order validated artifact `c584cc97…` on the local seed, every fixed-order gate green; spectral-order convergence is not certified).
- [x] Final `iota` within threshold: 0.049800 vs target 0.05 — error 2.0e-4 against the 2.5e-3 gate.
- [x] `metadata["solve_grid_points"]` = **1369** — updated policy: `(6·order+1)²` since `88879d1`, because the original 625-point `(4·order+1)²` grid was measured exploitable (E5f). 4096 remains wrong; 625 is now also wrong.
- [x] Rejected-trial fraction vs the same-seed unfixed baseline: **0 %** (E5i, 10 iterations) vs 42.2 % (E2 baseline).
- [x] Per-term objective breakdown logged on the real 368-coil set (2026-07-29): the current term dominates — ≥ 0.9999997 share under both weight sets; `BoozerResidual` share ~1.3e-10. Ablation plan, Results E1.
- [x] **Polarity equivalence** (2026-07-29): measured at both polarities on the local seed — signed `G` strictly reversed on every trace event while |iota|, |G|, J agree *exactly* (bit-exact mirror, 69 events). Ablation plan, Results E6. Caveat: measured through the instrumented harness at the current driver configuration; re-run through the fixed driver's publication gates once Phases 0/3 land.

Cluster (`/burg-archive`) — needed only for the exact `TF_a = 0.600` case:

- [ ] Reproduce the ablation result at Tian's `TF_a` if an absolute comparison to job 9209252 is required.
- [ ] Measured wall clock recorded here. **No speedup is claimed until this line is filled in.**

## Risks and Mitigations

- **Risk:** the unpushed local commits are a single point of failure. They exist on one
  machine, are unreviewed, and contain every fix this plan credits as done.
  **Mitigation:** Phase 1 pushes them before any further work. Treat "fixed on local `main`" as undelivered until then.
- **Risk:** Rebasing `tian-minimal` onto local `main` rewrites `4fee61e`, which is already
  pushed to `fork` and is the head of open PR #1.
  **Mitigation:** Land PR #1 first, then open the curated Stage-2 PR on top of the updated remote base (Phase 1). Do not force-push the PR head without asking. Note the two file sets are **not** disjoint — they share `boozer_functions.py`, `single_stage_dipoles.py`, `single_stage_dipoles.sh` — so ordering matters.
- **Risk:** Bounding the outer step slows convergence — a too-tight bound trades 45% rejections for many small accepted steps, with no net gain.
  **Mitigation:** Make the bound tunable; record accepted-step-norm distribution in the journal; compare total inner iterations, not just rejection rate.
- **Risk:** The iota-collapse → self-intersection causal link is assumed. If the real cause is elsewhere, fixing weights will not reduce rejections.
  **Mitigation:** Phase 6 measures the rejected fraction directly. If it stays near 45% after the weight fix, the assumption is falsified and the analysis must be reopened.
  **Outcome (measured 2026-07-29):** it stayed at ~40 % under both weight sets — weight
  changes alone demonstrably do not reduce rejections (ablation Results E2). The analysis
  is reopened exactly as this line required: the remedy is Phase 0 normalization (the
  current term dominates both weightings — E1), and the self-intersection link itself
  remains untested at this seed (local rejections are all Newton non-convergence).
- **Risk:** Omitting the scientific iota target silently changes the problem.
  **Mitigation:** Direct driver calls require `--iota-target`; historical comparisons must state it.
- **Risk:** The two `rejected_trial_value_and_gradient` copies have already drifted.
  **Mitigation:** Phase 2 step 1 diffs them before deduplicating.

## Completion Criteria

- [ ] The stage2-bridge commits are pushed to `fork` and open as a PR against `tianlangg:main`. Until then the fixes are undelivered regardless of local test status.
- [ ] That branch contains the archive commit, all correctness fixes, the step bound, and the deduplicated rejection model.
- [ ] Full test suite green; every new test above exists and passes.
- [ ] A cluster run on `TF_a = 0.600` publishes an artifact through the gates.
- [ ] Measured before/after numbers recorded in this file, sourced from `PhaseJournal`.
- [ ] Both open design questions answered by Tian and reflected in code or documented as deliberate.
- [ ] PR #1 updated or superseded.

## Open Questions

- **TF coil DOF scope** — owner: Tian. Should `stage2_to_single_stage.py:132` free all TF curve DOFs, or only radial translation/rotation? Raised by external review; unresolved and identical across `main` and the archive.
- ~~**iota target**~~ — **RESOLVED 2026-07-29:** at
  `--iota-target 0.3` the fixed driver runs 150/150 iterations and drives iota to 0.0006 —
  a ~10×-out-of-reach target saturates the iota penalty (max 0.5) and the current term's
  available 0.84 wins, so 0.3 reproduces a current-collapse run even with every fix in
  place. At the production 0.05 the outer optimization converges with iota ≈ target.
  Direct driver calls now require `--iota-target`; the Ginsburg launcher explicitly
  passes 0.05, avoiding a hidden scientific default.
- ~~**NEW LIVE DEFECT — the 25×25 solve grid is exploitable.**~~ **FIXED 2026-07-29,
  commit `88879d1`.** Measured chain: the E5e run's reported iota 0.0453 was really
  **0.0088** on a trustworthy grid (ablation E5f); at the certification's own
  `(6·order+1)²` density the refinement reproduces the solve state bit-exactly
  (E5g), closing the exploit; grid tax measured minor (E4c). The certification
  gate is what caught this — it was doing its job.
- ~~**Weight/target policy is the last gap to a published artifact**~~ **RESOLVED
  2026-07-29, commit `5e2c27c` — by fixing the term's semantics, not by tuning
  weights.** The raw p-norm made 150 kA a goal. As a one-sided soft-threshold
  penalty (zero below the threshold) the driver reaches iota
  0.049800 (error 2.0e-4 vs the 2.5e-3 gate) in 10 outer iterations with zero
  rejected trials and publishes a **fixed-order validated** artifact end-to-end
  (local-only ablations E5h/E5i; digest `c584cc97…`). The p-norm is 152.0 kA;
  that 1.3 % excess is allowed because 150 kA is explicitly a soft objective
  threshold, not a hard publication cap. Tian still owns the threshold value.
- **How should the archive be frozen?** — owner: user. `d79cc76` reordered
  `stage2_to_single_stage_tian_9209252.py` from 297 lines to 337, so the
  worktree copy is no longer what Tian ran. Either restore the archived file, or keep
  the reordered version under a clearly different name and point all forensic citations
  at `d610cdc`. **Until this is decided, cite `d610cdc`, never the worktree path** —
  citing the worktree is what produced the retracted accusation in the appendix.
- **What are the reference scales?** — owner: Tian. Partly answered by precedent, so what
  remains is narrower than "pick some numbers":
  - **Current:** `single_stage_dipoles.py` normalizes by `FCP_THRESHOLD` — the current
    p-norm limit, in amperes (`_outer_coordinate_scales`, `:489`/`:1335`/`:527`). The
    *form* is settled; only the *value* for this bridge is a physics-policy call, and it
    should track the intended current reference/limit. **Do not invent it silently.**
    Data points, not a recommendation: the local seed's Stage-2 solution has
    `max_wp_current = 110,990.70 A` and `current_threshold = 1e6 A`; production paths use
    `fcp150kA`. Any of these could be the intended reference — that is Tian's call.
  - **Iota:** `single_stage_dipoles.py:337` derives `IOTA_SCALE` from `--iota-target`;
    the bridge has no equivalent and needs one.
  - **Geometry:** if TF curve DOFs stay free (see TF DOF scope), a length scale is
    required and **no repo precedent exists** — this is the genuinely open one.
- ~~**Is `dipole_poloidal_rows = 10` correct, or is the local seed non-canonical?**~~
  **Resolved 2026-07-29:** the user ruled the seed canonical; `coil_layout.py` now says 11
  (commit `b367f36`). Flag to Tian on delivery — if he rules the opposite, revert that
  commit; every ablation measurement is void in that case (they all consumed the 352-dipole
  seed).
- **Was the G symptom ever real?** — owner: Tian. The polarity defect is proven in code, but it was never confirmed that this is what Tian actually observed. Carried over from `.handoffs/tian-single-stage-poster.md`.
- **Does `--sparse` / resume still work** after these changes? No execution receipt exists for either path.

## Appendix: notes on the two external review passes

Both passes were largely correct and materially improved the analysis. Recorded here so
the corrections are not relitigated:

- The runtime hours split (7.77 h / 3.72 h) was **derived** from the iteration split, not
  measured. Only the 67.6%-of-iterations figure is a measurement.
- `bfgs_tol=1e-10` / `newton_tol=1e-11` are tight tolerances, not machine precision — but
  the *achieved* residual was machine precision (median 1.82e-15), because Newton's
  quadratic convergence takes one step from 1e-10 to 1e-15.
- "Move the self-intersection check earlier" is not a valid remedy; see Phase 4.
- "It optimized a circular torus throughout" is wrong — the 148 surface DOFs move freely
  during the solve. Precise statement: it solved for **≈ the s = 0.55 flux surface**
  while reporting results as the s = 0.90 boundary.
- Review pass 2's finding #4 (unbounded outer steps) was initially dismissed as a symptom
  in analysis. It is the correct remedy and is now Phase 3.
- ~~Review line-number citations do not resolve.~~ **RETRACTED 2026-07-28 — this was my
  error, not theirs.** I checked their citations against `d610cdc` (297 lines) and the
  original download, never against the worktree file that `d79cc76` reordered into 337
  lines. Against the file they were actually reading, **every citation is exact**:

  | cited | line in the 337-line worktree file |
  | --- | --- |
  | `:294` repeated nested optimization | `res = boozersurface.run_code(run_dict["iota"], run_dict["G"])` |
  | `:123` surface fit discarded | `intermediate_surface = SurfaceXYZTensorFourier(` |
  | `:311` broken rejected-trial response | `if solve_success:` |
  | `:99` G convention | `sign_g = -1` |
  | `:45` iota target | `init_iota_guess = 0.3` |
  | `:52` unnormalized penalties | `NONQS_WEIGHT = 1.0` |
  | `:110` TF geometry free | `for coil in tf_coils:` |

  I twice told the user these were "unusable," "internally inconsistent," and likely
  fabricated. That was wrong and was caused by verifying against the wrong revision of a
  file I had been told was frozen. It was not frozen — which is finding 4 above.

**My own methodological error (2026-07-28).** The entire first analysis was run against
`/home/jungdaesuh/code/opensource/simsopt` (upstream `master`), not the pinned
`simsopt-dipoles` fork the repo actually uses. That install lacks both `CurrentPenalty`
and `ground`, so it cannot even import these drivers. Every load-bearing number was
re-verified against the correct build and **all reproduced exactly**: 3/148 default DOFs,
R 0.9000–1.1000, volume 1.973921e-01, Stage-2 0.326705 (60%), the s ≈ 0.55 bracket,
`QuadraticPenalty = 0.5·diff²`, `bfgs_tol=1e-10` / `newton_tol=1e-11` /
`bfgs_maxiter=1500`, `is_self_intersecting(angle=0.0)`, and `run_code`'s bare `return`.
No conclusion changed — but the checks had been running on the wrong interpreter, which
is why the header now pins one.

**Corrections from the doc-review pass (2026-07-28).** Reviewing this plan against the
repo found six errors in its own first draft:

1. *Critical.* It presented the fixes as "already fixed on `main`" without stating that
   `origin/main` is `49d47d7` and every commit is unpushed. A reader would have
   concluded the bridge work was delivered. It is not, and PR #1 does not contain it.
2. *Major.* Branch divergence was given as "9 / 2 commits"; actual is 8 on
   `tian-minimal`, 40 on local `main`, with merge-base `49d47d7` (Initial commit).
3. *Major.* The cluster check expected `solve_grid_points = 357`. The maintained driver
   now runs `RESOLUTION = 6` → 1,369. The earlier 357/625 figures belong to the
   retired 4× policy.
4. *Major.* "The objective was ~100% the current penalty" was stated as measured. The log
   prints only total `J`; `BoozerResidual`'s magnitude is unrecoverable from it. Downgraded
   to an argued inference with the gap named.
5. *Minor.* "Newton converged in 1–2 iterations on all 42 solves" is false — `{0: 1, 1: 37, 2: 4}`.
6. *Minor.* `s ≈ 0.55` → `s ≈ 0.56` (interpolated 0.557).

**Corrections from review pass 3 (2026-07-28).** All six of its blocking findings were
verified against the repo and all six were correct:

1. *Critical.* Unit weights do **not** fix iota starvation — current/iota is still 2.28e6
   because `CurrentPenalty` is in amperes. Added as **Phase 0**; the table row is now
   marked NOT FIXED. This is a live code defect, not a doc error.
2. *Critical.* `minimize_bounded_bfgs` bounds a raw L2 norm over DOFs mixing amperes and
   metres. Phase 3 now requires normalization first.
3. *Major.* Pushing local `main` wholesale is two dozen files and ~+7.8k lines. Phase 1
   now mandates a curated branch cut from `origin/main`.
4. *Major.* The archive is not frozen (`d79cc76` reordered it 297 → 337 lines).
5. *Major.* Polarity validation covered one sign only; equivalence check added.
6. *Major.* The "no term exceeds 99%" test is invalid — at the iota target that term is
   exactly zero. Replaced with normalization-formula tests.

Plus: Phase 4 renamed (telemetry, not a mechanism); the rejection-helper home reversed to
`bounded_bfgs`; the iota-collapse chain re-softened to a hypothesis; the half-period
domain cleared as legitimate (`Surface.RANGE_HALF_PERIOD`).

**Corrections from review pass 4 (2026-07-28).** All six verified; all six correct:

1. *Major.* "`stage2_to_single_stage.py` does not exist on `tian-minimal`" was false — it
   is present (282 lines, Tian's unfixed original). Caused by `grep -c … || echo ABSENT`.
2. *Major.* The MERGEABLE explanation was incoherent: GitHub evaluates against
   `origin/main` and cannot see unpushed commits. The file sets also overlap in three files.
3. *Major.* The curated file list omitted `bounded_bfgs.py`, which Phase 3 requires and
   which lives only in PR #1. Sequencing after PR #1 is now the default.
4. *Major.* The step-bound test asserted only on accepted trials — the case that never cost
   anything. Now every evaluated trial.
5. *Minor.* "Byte-exact" was literally false (one trailing blank line).
6. *Minor.* Pinned commit counts went stale on this plan's own commits. Counts are no
   longer pinned; the doc cites the command instead.

**Corrections from review pass 5 (2026-07-28).** One finding, correct — plus three the
sweep it prompted turned up:

1. *Major (theirs).* Phase 1 still carried the corrected-elsewhere claim that PR #1 "is
   MERGEABLE today only because its file set does not overlap local `main`". Fixing a
   duplicated claim in one location and not the other is its own failure mode; the fix now
   greps every occurrence rather than the cited line.
2. *Major (swept).* The Risks section referenced "Phase 1 option (a)" — the lettered
   options were deleted in the pass-3 rewrite, leaving a dangling cross-reference — and
   called the two PRs "disjoint", which pass 4 had already disproven.
3. *Minor (swept).* Phase 5 said "the two open design questions" while Open Questions
   listed six.
4. *Substantive (swept).* Phase 0 and Phase 3 were written as open design problems when
   `single_stage_dipoles.py` already ships the pattern: currents normalized by
   `FCP_THRESHOLD` via `_outer_coordinate_scales` (`:489`/`:1335`/`:527`), with every coil
   curve fixed (`:1971`, `:2007`) so its outer vector is homogeneous — which is exactly
   why its L2 step bound is meaningful. That reduces `I_scale` from "invent something" to
   "confirm a policy value", and gives Phase 3's route (b) a working precedent while
   route (a) has none.

**Correction from 2026-07-28 (local seed).** The plan asserted as a **FACT** that Stage-2
inputs were unavailable locally and that end-to-end validation was cluster-only. False.
The complete `bs_opt.json` / `surf_opt.json` / `results.json` contract sits one directory
above the checkout. The search that "proved" absence was run inside the checkout, while
`run_local_stage2_case.py`'s own docstring says the output is kept *outside* it. This
wrongly scoped Phase 6 and half the Validation Plan to a cluster for several revisions.

**The recurring failure this exposes.** Four separate false negatives this session came
from one habit — running a check, getting a negative, and reporting it without asking
whether the check itself could produce a false negative:

| # | probe | wrong conclusion |
| --- | --- | --- |
| 1 | upstream `simsopt` instead of the pinned fork | every numeric check run on the wrong build |
| 2 | line numbers against `d610cdc`, not the reordered worktree | called a reviewer's citations fabricated |
| 3 | `grep -c … \|\| echo ABSENT` | declared a file missing that exists |
| 4 | `fd` scoped to the checkout | declared the Stage-2 seed unavailable |

Pinning the interpreter, citing `d610cdc`, and using `git cat-file -e` fixed symptoms
1–3 individually. The habit is the defect: **a negative result is not evidence until the
probe is shown able to return a positive.** Phase 6 now encodes this as a step — validate
the harness against a known-correct value before trusting any new measurement.
