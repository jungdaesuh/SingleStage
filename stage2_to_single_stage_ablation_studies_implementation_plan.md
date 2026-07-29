# Single-Stage Optimizer: Ablation Studies (Stage2 → single-stage bridge)

**Status:** Done — all experiments E0–E6 measured 2026-07-29 (E5 unblocked same day by the user's `I_scale` = 150 kA decision and the Phase 0/2/3 implementation)
**Last updated:** 2026-07-29

> **What is under test.** Every experiment here varies the **single-stage** optimization —
> its objective terms and weights, its rejected-trial model, its Boozer trial solver, its
> quadrature grid, its outer step bound, and field polarity. **Stage 2 is a fixed input,
> never a variable.** Its converged output (`bs_opt.json` / `surf_opt.json` /
> `results.json`) is the seed every variant starts from, byte-identical across the whole
> ablation set — that is exactly what makes the deltas attributable to the single-stage
> changes. "stage2" appears in the filename only because the driver under test is
> `stage2_to_single_stage.py`.

> **Companion document.** This plan executes the measurements that
> [`stage2_to_single_stage_fix_implementation_plan.md`](stage2_to_single_stage_fix_implementation_plan.md)
> defers to its Phase 6. Read that file first — it owns the defect list, the
> post-mortem facts, and the local-seed description. This file owns only the
> experiments. Do not duplicate claims between them; cite the fix plan instead.

> **Interpreter.** `/home/jungdaesuh/code/columbia/dipole/tian/simsopt-dipoles/.venv/bin/python`.
> `CurrentPenalty` is fork-only; every driver here raises `ImportError` under upstream simsopt.
> Prefix long runs with `HWLOC_COMPONENTS=-gl` — without it, hwloc/GL topology enumeration
> can deadlock on `/dev/nvidia0`.

## Purpose

The 9209252 post-mortem rests on two claims that were argued rather than measured:

1. The objective is dominated by the current term (`BoozerResidual`'s share is unknown).
2. The iota collapse causes the self-intersections, hence the 45% rejection rate.

Both are labelled unproven in the fix plan. A local Stage-2 seed exists, so both are
now testable. This plan converts them into measurements and quantifies what each fix
actually buys — so that when the single-stage work is delivered to Tian, the claims attached
to it are receipts rather than reasoning.

## Goals

- The per-term objective split is **measured** on the real 368-coil seed, closing the
  `BoozerResidual` gap.
- The iota-collapse → self-intersection hypothesis is either confirmed or falsified by a
  controlled A/B, not inferred from one log.
- Each fix's individual contribution to rejection rate and inner-iteration count is
  quantified against a same-seed baseline.
- Every number produced here carries its scope limit, so none can be misread as a
  reproduction of job 9209252.

## Non-Goals

- Reproducing job 9209252 numerically. The seed is `TF_a = 0.4`; Tian ran `0.600`.
- Running anything on `/burg-archive`. Everything here is local.
- Implementing Phase 0 / Phase 3 fixes. Those belong to the fix plan; this plan measures
  them once they exist, and measures the current state before they do.
- Modifying `stage2_to_single_stage_tian_9209252.py`. It is forensic evidence.

## Current Context

### The seed (verified 2026-07-28)

```
SEED=/home/jungdaesuh/code/columbia/dipole/tian/outputs/stage2_initial_conditions/\
wout_nfp22ginsburg_000_001242/02_ntf4_diprad_0.05_VVa_0.26_VVb_0.27_VV_R0_1.03
```

Full parameters are tabulated in the fix plan under "Local Stage-2 seed". The facts this
plan depends on: `TF_a = 0.4`, `surf_nfp = 2`, `ntf = 4` (→ 16 TF coils), `num_wps = 352`,
`npoloidal = 11`, `ntoroidal = 8`, dipole currents 179.10 … 110,990.70 A.

The sibling directory `01_ntf4_...` contains only `optimize.py` and `helper_functions.py` —
no artifacts. **`02_` is the only usable seed.**

### Path shim — required, and verified to resolve

The driver computes `folder = base_dir / f"TF_a_{TF_a_cur:.3f}"` (`stage2_to_single_stage.py:76`)
and parses the value back out of the directory name (`:77`). The seed directory is not named
that way, so stage it:

```bash
SP=/tmp/claude-1000/-home-jungdaesuh-code-columbia-dipole-tian-SingleStage/\
d131180a-2580-4934-ad85-0a21bbec15f5/scratchpad
mkdir -p "$SP/ablation-root/TF_a_0.400"
for f in bs_opt.json surf_opt.json results.json; do
  ln -sf "$SEED/$f" "$SP/ablation-root/TF_a_0.400/$f"
done
```

Then invoke with `0.400 --stage2-root "$SP/ablation-root"`. Symlinks are fine: the driver
reads the files for its SHA-256 manifest (`:340-341`), which resolves through them.

### BLOCKER — RESOLVED 2026-07-29

**Resolution:** the user decided (2026-07-29) that the seed is canonical.
`CoilLayout.dipole_poloidal_rows` is now `11` (commit `b367f36`), so
`layout_for_nfp(2).dipole_physical_count` = 352 and the gate passes on the seed. The
toroidal identity `2 × wp_ntor × nfp = 32` is unchanged and remains the validated
invariant; the gate itself was not widened. Test expectations updated
(`tests/test_coil_layout.py`: 352 / 368); the 85-test suite passed before and after.

Original record, kept for history: `stage2_to_single_stage.py:117` raised
`ValueError("Stage-II dipole count disagrees with coil-layout SSOT")`.

| source | dipole count |
| --- | --- |
| seed `results.json` `num_wps` | **352** = 32 toroidal × **11** poloidal |
| `layout_for_nfp(2).dipole_physical_count` | **320** = 32 toroidal × **10** poloidal |

Root cause: `CoilLayout.dipole_poloidal_rows` is a bare hardcoded default of `10`
(`coil_layout.py:14`). It is not derived from anything and has no `nfp` dependence, so the
SSOT cannot express this seed. The toroidal side matches exactly
(`2 × wp_ntor(8) × nfp(2) = 32`), and the TF gate passes (16 = 16). **Poloidal rows are the
only mismatch.**

This is a genuine defect in its own right, not merely an ablation inconvenience: a "single
source of truth" that hardcodes one of its two dimensions will reject any valid
configuration that differs. Recorded here; ownership belongs to the fix plan.

## Rationale

Ablations run against the **same seed**, not against the 9209252 log. Absolute values from
`TF_a = 0.4` are not comparable to `TF_a = 0.600`, but deltas within one baseline are
exactly what an ablation needs. This sidesteps the cluster entirely.

Experiments are ordered cheapest-first and by information gained per unit cost. E1 needs no
driver run at all and kills a standing inference. E2 is the only experiment that can
*falsify* the post-mortem's central causal claim, so it runs before anything that assumes it.

Every harness validates against a known-correct value before its results are trusted. This
is not ceremony: four false negatives in this work came from probes that were never shown
able to return a positive (fix plan appendix). The `G` identity below is the cheapest
available ground truth.

## Assumptions

- **ASSUMPTION:** resolving the poloidal-rows blocker does not change the physics under
  test. Making `dipole_poloidal_rows` seed-derived changes only a validation gate. If it is
  instead resolved by regenerating a 10-row seed, the baseline changes and every prior
  measurement is void.
- **ASSUMPTION, now half-confirmed (2026-07-29):** a `TF_a = 0.4` seed exercises the same
  failure modes as `0.600`. Confirmed for the iota collapse, the ~40 % rejection regime,
  and the precision-loss line-search death (all reproduced — Results E2). **Not** confirmed
  for the rejection mechanism: local rejections are all Newton non-convergence, none
  self-intersection. Claims about the self-intersection pathway remain `0.600`-only.
- **FACT:** the seed's Stage-2 solve converged (`success: True`,
  `final_squared_flux = 6.373e-05`), so it is a valid starting point.
- **MEASURED (was UNKNOWN):** runtime per ablation at this seed — 24–93 min per
  reduced-`maxiter` run (Results table); a single `--maxiter 1` driver pass is 85.8 s.

## Implementation Plan

0. **Unblock and calibrate** *(prerequisite for everything below)*
   - [x] Resolve the poloidal-rows blocker. Resolved differently than the preferred option: the user ruled the seed canonical, so the SSOT default itself was corrected to `11` (`b367f36`) rather than made seed-derived. The gate and the toroidal invariant are untouched. Recorded in the fix plan, which owns `coil_layout.py`.
   - [x] **Harness ground truth.** PASS: derived signed `G` = −3.193408 vs expected |G| = μ₀ · 16 · 158,827.070125 A = 3.193408, relative error **1.3e-10** (`ablation_runs/e0_g_identity.py`). The seed's TF current is identical to job 9209252's — `field_on_axis` fixes it across `TF_a`, as predicted.
   - [x] Build the path shim above; all three files resolve and `TF_a` parses to 0.4.
   - [x] **Time one solve.** Driver at `--maxiter 1`: **85.8 s wall** end to end (init BFGS 333 iters 21.4 s + Newton 1.7 s; 3 outer-trial Newton solves at 0.13 s / 33.0 s failed / 12.6 s; rest is load overhead). A non-converging trial costs the full 40-iteration Newton budget (~33 s). Budget set from this: maxiter 12 for trajectory runs, 5 for E4b, 3 for E4c.

1. **E1 — per-term objective split** *(no driver run; settles fix-plan open item #10)*
   - [x] Loaded the seed's 368 coils, built the four terms as `stage2_to_single_stage.py:196-205` does, evaluated at the initial Boozer solution (`ablation_runs/e1_term_split.py`).
   - [x] Absolute values and shares recorded — see Results E1.
   - [x] Both weight sets on record (unit and archive `1e-2 / 100 / 100`).
   - [x] **Outcome: the current term dominates — CONFIRMED.** `BoozerResidual × 100` = 1.6e-3, nowhere near O(10⁴); its share is ~1.3e-10 under both weightings. The post-mortem's composition claim stands, now measured.

2. **E2 — does starving iota cause the collapse?** *(the falsification test)*
   - [x] Baseline run: unit weights, polarity −1, maxiter 12 (`ablation_runs/e2_baseline/`).
   - [x] Variant: archive weights `IOTAS_WEIGHT=1e-2, CURRS_WEIGHT=100, BZRES_WEIGHT=100`, maxiter 20 (`ablation_runs/e2_archive_weights/`).
   - [x] Rejected fraction, reasons, iota trajectory, inner iterations, wall clock recorded from each run's journal + trace — see Results E2.
   - [x] **Verdict against the pre-registered criterion: the archive-weighted variant does NOT show an elevated rejection rate or faster iota collapse relative to baseline — the hypothesis as stated is falsified.** Both arms collapse identically (see Results E2). Recorded verbatim, not reinterpreted; E1's measured composition (current ≥ 99.9997 % under *both* weightings) is the separate, consistent explanation.

3. **E3 — surface-fit regression, reproduced on demand**
   - [x] Fit-suppressed variant run (`ablation_runs/e3_deadfit.py`, report in `ablation_runs/e3/`).
   - [x] Confirmed: 3-of-148-DOF default torus (3 of 253 at RESOLUTION = 6), `targetlabel` = 0.19739208802178634 m³ — matches the analytic default-torus volume 2π²·R·a² to 8.6e-16 — vs the seed's 0.32670472159209635 m³ (ratio 0.6042). See Results E3 for the s-label caveat.
   - [x] **Purpose met:** the solver *converges cleanly* (Newton residual ~1e-14) onto the wrong surface, which is exactly why the defect was silent. Live reproduction available for a permanent regression test (open question below).

4. **E4 — per-fix contribution** *(each fix toggled off from the maintained-driver configuration)*
   - [x] Rejection model `J = prev_J, dJ = -prev_dJ`: **3 outer steps** before precision-loss death (baseline: 4) — the same count job 9209252 achieved; rejection rate 51 % vs 42.2 %.
   - [x] Full BFGS+Newton per trial: **424 inner iterations per trial vs 25.6** for Newton-only — a 16.6× inner-work tax. See Results E4.
   - [x] 64×64 grid (4,096 points): **1.16× median per-solve wall** vs 625 points — far below the 6.55× point ratio. See Results E4 and its unmeasured-cell caveat.
   - [x] All reported as deltas from the same-seed baseline with scope limits — see Results E4.

5. **E5 — the (now implemented) fixes** *(unblocked 2026-07-29: user set `I_scale` = 150 kA; Phases 0/2/3 landed as `d7f925a`, `f04e14c`, `d4a152b`)*
   - [x] **E5a — normalization alone** (`ablation_runs/e5a_normalized/`): the iota collapse is gone — accepted-trial iota drifts *up* (0.0274 → 0.0287) — but scipy BFGS accepts **zero** outer steps (precision loss at iteration 0, 56 trials, 34 % rejected). Normalizing the objective without normalizing the coordinates leaves the amperes+metres outer vector unusable, exactly as the fix plan warned.
   - [x] **E5b — normalization + currents-only outer vector** (`e5b_norm_currents_only/`): 12/12 outer iterations, monotone descent J 1.253 → 0.909, no precision-loss exit — the first run in this investigation to exhaust its budget instead of dying. Route (b) validated by measurement.
   - [x] **E5c — full fixed driver, default `--iota-target 0.3`** (`e5c_fixed_driver/`): 150/150 accepted steps, J 1.25 → 0.679 — and iota driven to 0.0006, because with a target ~10× beyond reach the saturated iota penalty (max 0.5) loses to the current term's available 0.84. Measured answer to the open iota-target question: **0.3 is wrong for this seed.** Fail-closed gates refused publication, correctly.
   - [x] **E5d/E5e — full fixed driver, production `--iota-target 0.05`** (`e5d_iota005/`, `e5e_publish/`): outer optimization **terminates successfully** — 148 accepted iterations, reported iota 0.0453, J 0.264, rejections 15 % (vs 42 % baseline). E5d exposed a latent driver bug (L2 post-gate against an inf-norm convergence contract; fixed in `d4a152b`). E5e then failed *downstream*, in certification: the 37×37 refinement Newton diverged.
   - [x] **E5f — refinement-divergence diagnosis** (`e5f/e5f_report.json`): probes on the saved converged state — fine-grid self-intersection **clean**; Newton-only refinement **diverges** (reproduces certification); LBFGS+Newton refinement **converges to the same volume and G but iota 0.0088, not 0.0453**. Verdict: at the then-current 25×25 solve grid (critical sampling for mpol=ntor=6), the outer optimizer raised its reported iota largely by **exploiting under-resolution** — the true Boozer surface at the optimized currents has iota ≈ 0.009. The certification gate caught a real defect. This historical defect was fixed by E5g and `88879d1`.
   - [x] **E5h — threshold current penalty** (`e5h_threshold.py`, `e5h/e5h_report.json`): replacing "minimize the p-norm" with the one-sided form `0.5·max(pnorm − 150 kA, 0)²/150 kA²` reaches **iota 0.049800 — 2.0e-4 from target — in 10 outer iterations**, converged, refinement probes bit-exact, with p-norm 152.0 kA and max 130.5 kA. The 1.3 % p-norm excess is allowed because 150 kA is a soft objective threshold, not a hard cap. Also answers attainability: iota 0.05 **is** reachable for this seed near that threshold.
   - [x] **E5i — the fixed driver publishes** (`e5i_publish/G_negative/TF_a_0.400_c584cc97978355d1/`): with the threshold form landed (`5e2c27c`), the real driver runs end to end through every fixed-order gate and publishes a fixed-order validated artifact — 10 outer iterations, **0 rejected trials**, 59.5 s total Boozer solve time; gates: iota error 2.0e-4 (≤ 2.5e-3), off-grid residual 4.7e-5 (≤ 1e-4), volume error 6.4e-4 (≤ 1e-3), refinement deltas exactly 0.0. Digest `c584cc97…`. Spectral-order convergence is explicitly not certified. The pipeline completes in ~10 minutes on this local seed.
   - [x] **E5j — a 4,096-point sensitivity run** (`e5j_grid64.py`, `e5j/e5j_report.json`): the order-6, full-field-period configuration reaches a nearby answer (iota 0.049695, 10 outer iterations, 0 rejections) in **3 min 0 s** wall vs ~2 min at 37×37. It shares the archive's point count only: the archive used `mpol=4`, `ntor=5`, a half-period domain, `TF_a=0.600`, and the old optimizer/objective. E5j therefore does not establish matched archive runtime or prove that grid cost was never dominant.
   - [x] **E5g — oversampled solve grid** (`e5g_dense.py`, (6·order+1)² = 37×37; `e5g/e5g_report.json`): **the exploit is closed** — both refinement probes (Newton-only and LBFGS+Newton) reproduce the solve state *bit-exactly* (iota 0.02381650004238853 in all three). Honest consequence: with truthful gradients the optimizer **cannot** raise iota toward 0.05 at unit weights — it settles at iota 0.0238 with the current p-norm down 36 % (126 → 81 kA), Armijo line search exhausting at 18 accepted iterations. The coarse grid's apparent iota progress was entirely quadrature artifact. Grid fix landed as `88879d1` (`build_boozer_solve_surface` now 6·order+1). **Remaining gap to a published artifact is weight/target policy — Tian's call**, recorded in the fix plan; not tuned here.

6. **E6 — polarity equivalence**
   - [x] Baseline configuration run at both polarities (`ablation_runs/e2_baseline/` at −1, `ablation_runs/e6_positive_polarity/` at +1).
   - [x] **PASS at zero tolerance.** Event-by-event trace comparison (69 events each): identical structure and accept/reject outcomes, max |Δiota| = 0.0, max ||G|−|G|| = 0.0, max relative ΔJ = 0.0, and `G`'s sign strictly reversed on every event (−3.193438 vs +3.193438). The two runs are bit-exact mirrors.

## Results (measured 2026-07-29)

**Scope limit on every number below: local seed, `TF_a = 0.4`, `--iota-target 0.3`,
`--initial-iota 0.045`, polarity −1 unless stated. E5d–E5j instead use the target stated
in their entries. Nothing here is a reproduction of job 9209252 (`TF_a = 0.600`).**
Raw artifacts were retained locally under `ablation_runs/` but are not part of commit
`5982139`; these measurements are therefore local receipts, not clean-checkout
reproduction evidence. `input_hashes` (SHA-256 of all three seed files) are identical
across the compared runs.

**Execution deviation, disclosed:** driver-based experiments ran through
`ablation_runs/harness.py`, an untracked, parameterized transcription of
`stage2_to_single_stage.py`'s flow (same contracts, same constants, same term
construction). Two metric-only differences: it appends a per-trial trace (iota/G/J and
reject reasons, which the driver's journal does not record), and it writes its summary
even when the outer BFGS fails the driver's `gtol` publication gate — which every
reduced-`maxiter` run does by construction. No physics knob differs from the driver at
baseline settings. Budgets were trimmed mid-queue (maxiter 20 → 12 for
baseline/E4a/E6) after the archive-weights arm showed ~40 rejected trials per line
search; A/B comparisons below are made at equal outer-step counts.

### E0 — harness ground truth and timing

- `G` identity: derived −3.193408 vs expected μ₀ · 16 · 158,827.070125 A = 3.193408;
  relative error **1.3e-10** — PASS. The seed's TF current equals job 9209252's exactly.
- One `--maxiter 1` driver run: **85.8 s** wall (init BFGS 333 iters / 21.4 s, Newton
  1 iter / 1.7 s, three outer trials 0.13 / 33.0 / 12.6 s). A non-converging trial burns
  the full `NEWTON_MAXITER = 40` (~33 s).

### E1 — per-term split at the initial Boozer solution (fix-plan open item #10: CLOSED)

Initial solve: iota 0.027436, G −3.193405, 625 grid points.

| term | raw value | share at unit weights | share at archive weights |
| --- | --- | --- | --- |
| `CurrentPenalty` (p=12, amperes) | **126,048.57** | **0.9999997** | **0.9999999998** |
| `QuadraticPenalty(Iotas, 0.3)` | 0.037146 | 2.9e-07 | 2.9e-11 |
| `NonQuasiSymmetricRatio` | 4.02e-05 | 3.2e-10 | 3.2e-12 |
| `BoozerResidual` | 1.62e-05 | 1.3e-10 | 1.3e-10 |

The current term dominates under **both** weightings; `BoozerResidual`'s previously
unmeasurable share is ~1.3e-10. Unit weights do not fix the starvation — the fix plan's
Phase 0 (normalization) is confirmed as the only real remedy.

### E2 — weight A/B (fix-plan open item #11: measured; hypothesis falsified as stated)

| arm | outer steps | trials | rejected | iota (init → end) | exit | wall |
| --- | --- | --- | --- | --- | --- | --- |
| unit weights (baseline) | 4 / 12 | 64 | 27 (42.2 %) | 0.027436 → 0.015248 | precision loss | 3,383 s |
| archive weights | 5 / 20 | 105 | 41 (39.0 %) | 0.027436 → 0.013876 | precision loss | 3,948 s |

At equal step counts the trajectories are near-identical (baseline steps 1–4: 0.026309,
0.022520, 0.019356, 0.015248; archive: 0.026309, 0.022519, 0.020397, 0.016719). Per the
pre-registered criterion — archive weights must show collapse *and elevated rejection
relative to baseline* — **the hypothesis is falsified**: the collapse is
weight-independent. Iota starvation itself is real in both arms (E1 explains why: the
current term dominates both compositions). Both arms end the same way 9209252 did:
tens of consecutive rejected trials in one line search, then scipy's
"Desired error not necessarily achieved due to precision loss".

**Scope note:** every rejection in every local run is "Boozer solver did not converge";
job 9209252's 19 rejections were all self-intersection. The rejection *mechanism*
differs between `TF_a = 0.4`/625 points and `TF_a = 0.600`/4,096 points; the collapse
and the line-search death spiral are what reproduce.

### E3 — dead-fit reproduction (live regression of the fatal defect)

- Unfitted solve surface: **3 nonzero DOFs** (of 253 at RESOLUTION = 6; of 148 at the
  seed surface's own Fourier order) — `x(0,0)=1.0, x(1,0)=0.1, z(7,0)=0.1`, the default
  torus. Volume 0.19739208802178634 m³ = analytic 2π²·R·a² to 8.6e-16.
- `targetlabel` therefore pins the solve to 0.6042 of the seed's volume; the solver
  **converges cleanly** (BFGS 368 iters + Newton 1 iter, residual ~1e-14) to iota
  0.028508, G −3.193401, volume 0.19736 m³ — plausible numbers on the wrong flux
  surface, which is why the defect survived 11.5 hours of production compute unnoticed.
- Label caveat: the measured **volume ratio is 0.604**; the post-mortem's **s ≈ 0.56**
  is the VMEC-profile interpolation of that volume (V(s) is nonlinear). Both stand,
  with derivations attached.

### E4 — per-fix deltas vs the same-seed baseline

| variant (one toggle each) | outer steps | rejected | inner iters | wall | delta that matters |
| --- | --- | --- | --- | --- | --- |
| baseline (all fixes on) | 4 / 12 | 42.2 % | 1,970 | 3,383 s | reference |
| E4a: legacy rejection `J=prev_J, dJ=−prev_dJ` | **3** / 12 | 51.0 % | 1,698 | 2,855 s | dies one step earlier — the same 3-outer-step count as job 9209252 |
| E4b: full BFGS+Newton per trial | 4 / 5 | 36.4 % | **28,324** | 5,593 s | **424 inner iters/trial vs 25.6 — a 16.6× inner-work tax**; the Newton-only warm start is the largest single saving measured |
| E4c: 64×64 grid (4,096 pts), maxiter 3 | 1 / 3 | 45.5 % | 844 | 1,422 s | median outer-Newton solve **59.1 s vs 51.0 s — 1.16×**, far below the 6.55× point ratio |

E4c caveats: (i) the 9209252 combination — full BFGS+Newton × 4,096 points — was **not
measured**; per-solve cost of that cell cannot be inferred from these two one-factor
runs. (ii) The wall-clock noise floor on this machine is ~10 % (E6, doing identical
work to baseline, has median solve 45.8 s vs 51.0 s). (iii) The E4c surface is fitted
correctly on its own grid, so only grid cost is under test; its targetlabel differs in
the 4th decimal (0.32666 vs 0.32670 m³) and its init iota by 4e-4. Conclusion that
survives the caveats: for the *Newton-only* trial path at this seed, quadrature-point
count is a minor cost factor; the 11.47× point ratio in the fix plan is a statement
about points, not time, and must not be quoted as a runtime factor.

### E6 — polarity equivalence: PASS at zero tolerance

69 trace events per run, identical structure and outcomes; |iota|, |G|, J agree
**exactly** (max differences 0.0) while `G`'s sign is strictly reversed on every event.
`apply_field_polarity` produces a bit-exact mirrored optimization on this seed.

### E5 — the fixes, measured (added 2026-07-29, after Phases 0/2/3 landed)

See the Implementation Plan's E5 entry for the per-step ladder. The headline sequence:
normalization alone → collapse cured but zero outer steps; + currents-only outer vector
→ monotone descent, no more precision-loss deaths; full fixed driver at the production
iota target → outer optimization converges (148 iterations, 15 % rejection) but
certification correctly rejects the result: the then-current 25×25 solve grid is
exploitable, and the
optimizer's reported iota 0.0453 is really 0.0088 on a trustworthy grid (E5f). One
latent driver bug found and fixed on the way (L2-vs-inf-norm gate, `d4a152b`).
E5g closed that defect: the maintained solve grid is now 37×37/1,369 points
(`88879d1`), with a separate 49×49 held-out residual grid.

### What was NOT measured

- The full-solver × 4,096-point cell (job 9209252's actual configuration).
- Any run at `TF_a = 0.600` or any self-intersection rejection. In E0–E4 every arm died
  of precision loss — itself the measured reproduction of 9209252's exit; E5d/E5e are
  the first gtol-met outer optimizations.

## Validation Plan

- [x] `HWLOC_COMPONENTS=-gl <venv>/bin/python -m pytest tests/ -q` passed before (2026-07-28) and after (2026-07-29) the `coil_layout.py` change — 85 tests both times.
- [x] E0's `G` identity reproduced to 1.3e-10 before any experiment ran.
- [x] Numbers read from `phase_records_TF_a_0.400.jsonl` plus the harness's `ablation_trace.jsonl` (added because the journal does not record iota/G/J per trial); no stdout scraping.
- [x] Historical E0–E5f `solve_grid_points` asserted per run via local
  `ablation_summary.json`: 625 for the retired 4× policy and 4,096 for E4c.
  E5g–E5i use the maintained 37×37/1,369-point policy.
- [x] `input_hashes` SHA-256s identical across all six runs (`consolidated_metrics.json`: `_seed_hash_identical_across_runs: true`).
- [x] E2's falsification criterion was pre-registered in this file (2026-07-28, before any run) and the outcome is reported against it verbatim in Results E2.
- [x] Every result in Results carries the `TF_a = 0.4` scope limit.

## Risks and Mitigations

- **Risk:** the poloidal-rows fix is made by loosening the gate to whatever the seed says, silently destroying the SSOT's purpose.
  **Mitigation:** derive the value from `results["npoloidal"]` and keep the toroidal identity `2 × wp_ntor × nfp = 32` as an enforced invariant. Changing a validation gate to accept the input it is validating is not a fix.
- **Risk:** runtime makes the full matrix infeasible — 6 experiments × multiple variants.
  **Mitigation:** E0 times one solve first. If a single run exceeds a few hours, cut to E1 + E2 only; those carry nearly all the information.
- **Risk:** an ablation number gets quoted against job 9209252's 45% or 11.5 h.
  **Mitigation:** `TF_a` differs. Every table in this plan carries the scope limit; the write-up gate above enforces it.
- **Risk:** E2 confirms the hypothesis for the wrong reason — iota collapses because of something else that the archive weights also perturb.
  **Mitigation:** E1 runs first and establishes the actual term shares, so E2's result is interpreted against measured composition rather than assumed dominance.
- **Risk:** measuring the current state and calling it "the fixed driver" when Phase 0 and Phase 3 are still unimplemented.
  **Mitigation:** E5 is explicitly marked blocked. The baseline in E2/E4 is the *partially* fixed driver, and must be described that way.

## Completion Criteria

- [x] The poloidal-rows blocker is resolved (`b367f36`) and the solve pipeline runs end to end on the local seed (outer optimizations terminate by precision loss, not by the SSOT gate — see the "not measured" list for what "end to end" does not include).
- [x] E1 reports the four-term split with `BoozerResidual`'s share measured (1.3e-10); fix-plan open item #10 closed — claim confirmed.
- [x] E2 reports against its pre-registered falsification criterion (hypothesis falsified as stated); fix-plan open item #11 closed.
- [x] E3 and E4 report per-fix deltas from a same-seed baseline.
- [x] E6 passes (bit-exact mirror, zero tolerance needed).
- [x] Results folded back into the fix plan's Phase 6 (2026-07-29).
- [x] Every reported number carries the `TF_a = 0.4` scope limit.

## Open Questions

- ~~**Is `dipole_poloidal_rows = 10` correct?**~~ **Resolved 2026-07-29:** the user ruled
  the seed canonical; SSOT set to 11 in `b367f36`. If Tian later rules the opposite, revert
  that commit and regenerate a 10-row seed — every measurement here would then be void
  (see Assumptions).
- **What is `I_scale`?** — owner: Tian. Tracked in the fix plan; blocks E5. Data points from
  this seed: `max_wp_current = 110,990.70 A`, `current_threshold = 1e6 A`.
- **Is a `TF_a = 0.4` ablation persuasive to Tian**, or does the write-up need the `0.600`
  case on the cluster? — owner: user. Sharpened by E2's scope note: the rejection
  *mechanism* differs at this seed (Newton non-convergence, not self-intersection), so the
  `0.600` case is the only way to reproduce the self-intersection pathway itself.
- **Should E3 ship as a permanent regression test** rather than a one-off ablation? It is
  the fatal defect; `ablation_runs/e3_deadfit.py` is a working reproduction that a
  `tests/` case could adapt (the DOF/volume assertions run in milliseconds without a
  Boozer solve).
