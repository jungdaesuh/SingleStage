"""Shared contracts for Tian's Stage-2 to single-stage bridge."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp, mkstemp
from time import perf_counter, time
from typing import Mapping

import numpy as np

from simsopt.field import CurrentPenalty
from simsopt.geo import (
    BoozerSurface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
)
from simsopt.objectives import QuadraticPenalty


MU0 = 4.0e-7 * np.pi
BFGS_TOLERANCE = 1.0e-10
BFGS_MAXITER = 500
NEWTON_TOLERANCE = 1.0e-11
NEWTON_MAXITER = 40
CERTIFICATE_POLICY_ID = "tian-stage2-fixed-order-v1"

# Objective-normalization scales. CurrentPenalty returns amperes, and
# BoozerResidual is a residual-squared objective rather than a ratio. Both
# therefore enter through normalized one-sided soft thresholds.
CURRENT_SCALE_AMPERES = 150_000.0
CURRENT_P_NORM = 12
BOOZER_RESIDUAL_THRESHOLD = 1.0e-4


def current_p_norm_amperes(currents, p: int = CURRENT_P_NORM) -> float:
    """Return the current p-norm in amperes."""

    return float(CurrentPenalty(currents, p=p).J())


def current_limit_penalty(
    currents,
    p: int = CURRENT_P_NORM,
    limit_amperes: float = CURRENT_SCALE_AMPERES,
):
    """Dimensionless one-sided soft current-threshold term.

    Exactly zero while the p-norm of ``currents`` (amperes) stays below
    ``limit_amperes``; 0.5*((pnorm - limit)/limit)**2 above it. The limit is a
    soft objective threshold, not a hard publication cap or minimization goal,
    so admissible currents feel no pull toward zero.
    """

    limit = float(limit_amperes)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"current limit must be finite and positive, got {limit_amperes}")
    return (1.0 / limit**2) * QuadraticPenalty(
        CurrentPenalty(currents, p=p), limit, "max"
    )


def boozer_residual_guard(boozer_residual):
    """Dimensionless one-sided guard at the fixed-order residual threshold."""

    return (1.0 / BOOZER_RESIDUAL_THRESHOLD**2) * QuadraticPenalty(
        boozer_residual,
        BOOZER_RESIDUAL_THRESHOLD,
        "max",
    )


def iota_penalty_curvature(iota_target: float) -> float:
    """Curvature 1/IOTA_SCALE**2 with IOTA_SCALE = |iota_target| (or 1.0 at 0).

    Scaling QuadraticPenalty(Iotas, target) by this makes an iota error of one
    scale contribute exactly 0.5, matching single_stage_dipoles.py's
    IOTA_WEIGHT / IOTA_SCALE**2 form.
    """

    target = float(iota_target)
    if not np.isfinite(target):
        raise ValueError(f"iota target must be finite, got {iota_target}")
    scale = abs(target) or 1.0
    return 1.0 / (scale * scale)


@dataclass(frozen=True)
class BoozerPhaseReport:
    phase: str
    success: bool
    elapsed_seconds: float
    iterations: int
    function_evaluations: int | None

    def to_dict(self) -> dict[str, str | float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class BoozerCertificate:
    offgrid_residual: float
    relative_volume_error: float
    iota_target_error: float
    iota_refinement_delta: float
    relative_G_refinement_delta: float
    refinement_nphi: int
    refinement_ntheta: int
    certificate_nphi: int
    certificate_ntheta: int
    fourier_mpol: int
    fourier_ntor: int
    spectral_order_certified: bool

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BoozerCertificateThresholds:
    max_offgrid_residual: float
    max_relative_volume_error: float
    max_iota_error: float
    max_iota_refinement_delta: float
    max_relative_G_refinement_delta: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


CERTIFICATE_THRESHOLDS = BoozerCertificateThresholds(
    max_offgrid_residual=BOOZER_RESIDUAL_THRESHOLD,
    max_relative_volume_error=1.0e-3,
    max_iota_error=2.5e-3,
    max_iota_refinement_delta=2.5e-3,
    max_relative_G_refinement_delta=1.0e-3,
)


class BoozerSolveError(RuntimeError):
    """Raised when a Boozer solve cannot be certified."""


class PhaseJournal:
    """Append-only JSONL journal for solver phases, including failed phases."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        phase: str,
        event: str,
        started_monotonic: float,
        success: bool | None = None,
        iterations: int | None = None,
        function_evaluations: int | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "phase": phase,
            "event": event,
            "wall_time": time(),
            "monotonic_time": perf_counter(),
            "started_monotonic": started_monotonic,
            "elapsed_seconds": max(0.0, perf_counter() - started_monotonic),
            "success": success,
            "iterations": iterations,
            "function_evaluations": function_evaluations,
            "error": error,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()


def create_phase_journal(
    output_directory: str | Path,
    *,
    TF_a: float,
) -> PhaseJournal:
    """Create one atomically unique journal for an optimization process."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    descriptor, filename = mkstemp(
        prefix=f"phase_records_TF_a_{TF_a:.3f}_",
        suffix=".jsonl",
        dir=output_path,
    )
    os.close(descriptor)
    return PhaseJournal(filename)


def check_self_intersection_angles(surface, *, nfp: int) -> tuple[float, ...]:
    """Run the finite, Nfp-scaled cylindrical-angle geometry screen."""

    if nfp <= 0:
        raise ValueError(f"nfp must be positive, got {nfp}")
    angles = tuple(
        (2.0 * np.pi / nfp) * fraction
        for fraction in (0.0, 0.25, 0.5, 0.75)
    )
    for angle in angles:
        try:
            self_intersecting = surface.is_self_intersecting(angle=angle)
        except Exception as error:
            raise BoozerSolveError(
                f"self-intersection certification failed at angle {angle:.16g}: {error}"
            ) from error
        if self_intersecting:
            raise BoozerSolveError(
                f"Boozer surface is self-intersecting at angle {angle:.16g}"
            )
    return angles


def validate_checkpoint_manifest(
    manifest_path: str | Path,
    *,
    expected: Mapping[str, object] | None = None,
) -> tuple[Path, dict[str, object]] | None:
    """Validate one atomic checkpoint generation and return its path/manifest."""

    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        return None
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    generation = manifest.get("generation")
    files = manifest.get("files")
    if not isinstance(generation, str) or not isinstance(files, dict):
        raise ValueError(f"invalid checkpoint manifest at {manifest_file}")
    if expected is not None:
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"checkpoint policy mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
    checkpoint_path = manifest_file.parent / "checkpoints" / generation
    if not checkpoint_path.is_dir():
        raise ValueError(f"checkpoint generation is missing: {checkpoint_path}")
    for filename, expected_hash in files.items():
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise ValueError("checkpoint manifest file entries must be strings")
        file_path = checkpoint_path / filename
        if not file_path.is_file():
            raise ValueError(f"checkpoint file is missing: {file_path}")
        actual_hash = sha256(file_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"checkpoint hash mismatch for {filename}: "
                f"{actual_hash} != {expected_hash}"
            )
    return checkpoint_path, manifest


def normalize_polarity(value: int) -> int:
    polarity = int(value)
    if polarity not in (-1, 1):
        raise ValueError(f"field polarity must be -1 or +1, got {value!r}")
    return polarity


def _base_tf_currents(tf_coils, *, ntf: int):
    if ntf <= 0:
        raise ValueError(f"ntf must be positive, got {ntf}")
    if len(tf_coils) < ntf:
        raise ValueError(
            f"TF bundle has {len(tf_coils)} coils but requires at least {ntf} base coils"
        )
    currents = np.asarray(
        [coil.current.get_value() for coil in tf_coils[:ntf]], dtype=float
    )
    if not np.all(np.isfinite(currents)) or np.any(currents == 0.0):
        raise ValueError("base TF currents must be finite and non-zero")
    signs = np.sign(currents)
    if not np.all(signs == signs[0]):
        raise ValueError("base TF currents must share one physical polarity")
    return currents


def derive_signed_G(
    tf_coils,
    *,
    ntf: int,
    nfp: int,
    stellsym: bool,
) -> float:
    """Derive signed G from the independent TF bundle.

    Symmetry-generated mirrored coils report opposite current signs because
    their curve orientation is reversed. Therefore the independent first
    half-period bundle is summed and multiplied by its symmetry multiplicity.
    """

    expected_coils = ntf * nfp * (2 if stellsym else 1)
    if len(tf_coils) != expected_coils:
        raise ValueError(
            f"TF bundle has {len(tf_coils)} coils; expected {expected_coils} "
            f"for ntf={ntf}, nfp={nfp}, stellsym={stellsym}"
        )
    base_currents = _base_tf_currents(tf_coils, ntf=ntf)
    symmetry_multiplicity = nfp * (2 if stellsym else 1)
    return float(MU0 * symmetry_multiplicity * np.sum(base_currents))


def _root_current(current):
    root = current
    while hasattr(root, "current_to_scale"):
        root = root.current_to_scale
    if not hasattr(root, "get_value") or not (
        hasattr(root, "set_value") or hasattr(root, "local_full_x")
    ):
        raise TypeError(
            f"unsupported current wrapper {type(current).__name__}: "
            "cannot locate a mutable root current"
        )
    return root


def _set_root_current_value(root, value: float) -> None:
    if hasattr(root, "set_value"):
        root.set_value(value)
        return
    root.local_full_x = np.asarray([value], dtype=float)


def apply_field_polarity(
    coils,
    tf_coils,
    *,
    polarity: int,
    ntf: int,
    nfp: int,
    stellsym: bool,
) -> float:
    """Globally reverse the loaded field if needed, then return signed G."""

    requested_polarity = normalize_polarity(polarity)
    current_G = derive_signed_G(
        tf_coils,
        ntf=ntf,
        nfp=nfp,
        stellsym=stellsym,
    )
    if int(np.sign(current_G)) != requested_polarity:
        roots = {}
        for coil in coils:
            root = _root_current(coil.current)
            roots[id(root)] = root
        for root in roots.values():
            _set_root_current_value(root, -float(root.get_value()))
        current_G = derive_signed_G(
            tf_coils,
            ntf=ntf,
            nfp=nfp,
            stellsym=stellsym,
        )
    if int(np.sign(current_G)) != requested_polarity:
        raise RuntimeError(
            f"failed to realize requested field polarity {requested_polarity:+d}; "
            f"derived G={current_G:.16g}"
        )
    return current_G


def build_boozer_solve_surface(
    source_surface,
    *,
    mpol: int,
    ntor: int,
) -> SurfaceXYZTensorFourier:
    """Build an oversampled field-period grid for the BoozerLS search."""

    # 6x, not 4x: (4*order+1) is critical sampling and the outer optimizer
    # exploits it — measured 2026-07-29 (ablation E5e/E5f): a run reported
    # iota 0.0453 whose true surface, re-solved at 6x density, has iota 0.0088.
    # At 6x the certification refinement reproduces the solve state exactly
    # (E5g) and the grid tax is minor (E4c: 1.16x per solve at 4,096 points).
    nphi = 6 * ntor + 1
    ntheta = 6 * mpol + 1
    quadpoints_phi = np.linspace(
        0.0, 1.0 / source_surface.nfp, nphi, endpoint=False
    )
    quadpoints_theta = np.linspace(0.0, 1.0, ntheta, endpoint=False)

    if isinstance(source_surface, SurfaceRZFourier):
        sampled_source = SurfaceRZFourier(
            mpol=source_surface.mpol,
            ntor=source_surface.ntor,
            nfp=source_surface.nfp,
            stellsym=source_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        sampled_source.x = source_surface.x.copy()
    elif isinstance(source_surface, SurfaceXYZTensorFourier):
        sampled_source = SurfaceXYZTensorFourier(
            mpol=source_surface.mpol,
            ntor=source_surface.ntor,
            nfp=source_surface.nfp,
            stellsym=source_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        sampled_source.x = source_surface.x.copy()
    else:
        raise TypeError(
            "source surface must be SurfaceRZFourier or "
            f"SurfaceXYZTensorFourier, got {type(source_surface).__name__}"
        )

    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=source_surface.nfp,
        stellsym=source_surface.stellsym,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.least_squares_fit(sampled_source.gamma())
    return surface


def prolong_boozer_surface(
    source_surface: SurfaceXYZTensorFourier,
    *,
    mpol: int,
    ntor: int,
) -> SurfaceXYZTensorFourier:
    """Lift a Boozer surface to higher order while preserving its geometry.

    The stellarator-symmetric coefficient blocks change layout when ``mpol`` or
    ``ntor`` changes, so raw array slicing is not a valid prolongation.  Fit the
    target coefficients to the source surface on the source quadrature grid;
    this preserves the represented geometry and leaves newly available modes at
    zero up to the numerical least-squares tolerance.
    """

    if mpol < source_surface.mpol or ntor < source_surface.ntor:
        raise ValueError("Boozer prolongation cannot lower Fourier order")
    target = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=source_surface.nfp,
        stellsym=source_surface.stellsym,
        quadpoints_phi=source_surface.quadpoints_phi.copy(),
        quadpoints_theta=source_surface.quadpoints_theta.copy(),
    )
    target.least_squares_fit(source_surface.gamma())
    return target


def _resample_surface(surface, *, nphi: int, ntheta: int):
    resampled = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        quadpoints_phi=np.linspace(0.0, 1.0 / surface.nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    resampled.x = surface.x.copy()
    return resampled


def _phase_report(phase: str, result, elapsed_seconds: float) -> BoozerPhaseReport:
    info = result.get("info")
    function_evaluations = None if info is None else int(info.nfev)
    return BoozerPhaseReport(
        phase=phase,
        success=bool(result["success"]),
        elapsed_seconds=elapsed_seconds,
        iterations=int(result["iter"]),
        function_evaluations=function_evaluations,
    )


def solve_boozer_initialization(
    boozer_surface,
    iota: float,
    G: float,
    *,
    phase_journal: PhaseJournal | None = None,
):
    """Run one bounded BFGS globalization followed by a Newton polish."""

    phase = "initial_bfgs"
    started = perf_counter()
    if phase_journal is not None:
        phase_journal.record(phase=phase, event="start", started_monotonic=started)
    try:
        bfgs_result = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=BFGS_TOLERANCE,
            maxiter=BFGS_MAXITER,
            constraint_weight=boozer_surface.constraint_weight,
            iota=iota,
            G=G,
            limited_memory=False,
            weight_inv_modB=True,
            verbose=True,
        )
    except Exception as error:
        if phase_journal is not None:
            phase_journal.record(
                phase=phase,
                event="end",
                started_monotonic=started,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    bfgs_report = _phase_report("initial_bfgs", bfgs_result, perf_counter() - started)
    if phase_journal is not None:
        phase_journal.record(
            phase=phase,
            event="end",
            started_monotonic=started,
            success=bfgs_report.success,
            iterations=bfgs_report.iterations,
            function_evaluations=bfgs_report.function_evaluations,
        )
    boozer_surface.need_to_run_code = True
    newton_result, newton_report = solve_boozer_newton(
        boozer_surface,
        float(bfgs_result["iota"]),
        float(bfgs_result["G"]),
        phase="initial_newton",
        phase_journal=phase_journal,
    )
    return newton_result, (bfgs_report, newton_report)


def solve_boozer_newton(
    boozer_surface,
    iota: float,
    G: float,
    *,
    phase: str,
    phase_journal: PhaseJournal | None = None,
):
    """Warm-start one bounded Newton solve and return its phase telemetry."""

    started = perf_counter()
    if phase_journal is not None:
        phase_journal.record(phase=phase, event="start", started_monotonic=started)
    try:
        result = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=NEWTON_TOLERANCE,
            maxiter=NEWTON_MAXITER,
            constraint_weight=boozer_surface.constraint_weight,
            iota=iota,
            G=G,
            weight_inv_modB=True,
            verbose=True,
        )
    except Exception as error:
        if phase_journal is not None:
            phase_journal.record(
                phase=phase,
                event="end",
                started_monotonic=started,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    report = _phase_report(phase, result, perf_counter() - started)
    if phase_journal is not None:
        phase_journal.record(
            phase=phase,
            event="end",
            started_monotonic=started,
            success=report.success,
            iterations=report.iterations,
            function_evaluations=report.function_evaluations,
        )
    return result, report


def solve_boozer_outer_trial(
    boozer_surface,
    iota: float,
    G: float,
    *,
    phase_journal: PhaseJournal | None = None,
):
    """Warm-start one outer-optimizer trial using Newton only."""

    boozer_surface.need_to_run_code = True
    result, report = solve_boozer_newton(
        boozer_surface,
        iota,
        G,
        phase="outer_newton",
        phase_journal=phase_journal,
    )
    return result, report


def restore_accepted_boozer_state(
    boozer_surface,
    joint_objective,
    *,
    optimizer_dofs,
    surface_dofs,
    iota: float,
    G: float,
) -> None:
    """Restore a solved accepted state without scheduling an implicit solve."""

    boozer_surface.surface.x = np.asarray(surface_dofs).copy()
    boozer_surface.res["iota"] = float(iota)
    boozer_surface.res["G"] = float(G)
    joint_objective.x = np.asarray(optimizer_dofs).copy()
    # Setting any dependent DOF invalidates the Optimizable graph and marks the
    # Boozer surface dirty.  The restored state is already the last solved,
    # accepted state, so a later diagnostic J()/dJ() read must not call
    # BoozerSurface.run_code() and silently reintroduce outer BFGS work.
    boozer_surface.need_to_run_code = False


def set_boozer_constraint_weight(boozer_surface, weight: float) -> None:
    """Change the Boozer label weight and invalidate the cached solve."""

    boozer_surface.constraint_weight = float(weight)
    boozer_surface.need_to_run_code = True


def require_valid_boozer_result(
    result,
    surface,
    *,
    expected_G_sign: int,
    nfp: int | None = None,
):
    """Return a certified result or raise with the exact failed contract."""

    if result is None:
        raise BoozerSolveError("Boozer solver returned no result")
    if not bool(result.get("success", False)):
        raise BoozerSolveError("Boozer solver did not converge")
    if nfp is None:
        try:
            self_intersecting = surface.is_self_intersecting()
        except RuntimeError as error:
            raise BoozerSolveError(
                f"self-intersection certification failed: {error}"
            ) from error
        if self_intersecting:
            raise BoozerSolveError("Boozer surface is self-intersecting")
    else:
        check_self_intersection_angles(surface, nfp=nfp)
    solved_G = float(result["G"])
    if not np.isfinite(solved_G) or int(np.sign(solved_G)) != normalize_polarity(
        expected_G_sign
    ):
        raise BoozerSolveError(
            f"Boozer G violated requested polarity: G={solved_G:.16g}, "
            f"expected sign {expected_G_sign:+d}"
        )
    solved_iota = float(result["iota"])
    if not np.isfinite(solved_iota):
        raise BoozerSolveError(f"Boozer solver returned non-finite iota={solved_iota}")
    return result


def certify_boozer_solution(
    boozer_surface,
    biotsavart,
    *,
    expected_G_sign: int,
    iota_target: float,
    phase_journal: PhaseJournal | None = None,
):
    """Re-solve and enforce independent gates at one fixed Fourier order."""

    coarse_result = require_valid_boozer_result(
        boozer_surface.res,
        boozer_surface.surface,
        expected_G_sign=expected_G_sign,
        nfp=boozer_surface.surface.nfp,
    )
    refinement_surface = _resample_surface(
        boozer_surface.surface,
        nphi=6 * boozer_surface.surface.ntor + 1,
        ntheta=6 * boozer_surface.surface.mpol + 1,
    )
    refinement_boozer = BoozerSurface(
        biotsavart,
        refinement_surface,
        Volume(refinement_surface),
        boozer_surface.targetlabel,
        boozer_surface.constraint_weight,
    )
    refinement_result, refinement_report = solve_boozer_newton(
        refinement_boozer,
        float(coarse_result["iota"]),
        float(coarse_result["G"]),
        phase="final_refinement_newton",
        phase_journal=phase_journal,
    )
    require_valid_boozer_result(
        refinement_result,
        refinement_surface,
        expected_G_sign=expected_G_sign,
        nfp=refinement_surface.nfp,
    )

    certificate_surface = _resample_surface(
        refinement_surface,
        nphi=8 * refinement_surface.ntor + 1,
        ntheta=8 * refinement_surface.mpol + 1,
    )
    residual_phase = "certificate_residual"
    residual_started = perf_counter()
    if phase_journal is not None:
        phase_journal.record(
            phase=residual_phase,
            event="start",
            started_monotonic=residual_started,
        )
    try:
        residual = boozer_surface_residual(
            certificate_surface,
            float(refinement_result["iota"]),
            float(refinement_result["G"]),
            biotsavart,
            derivatives=0,
            weight_inv_modB=True,
        )[0]
    except Exception as error:
        if phase_journal is not None:
            phase_journal.record(
                phase=residual_phase,
                event="end",
                started_monotonic=residual_started,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    if phase_journal is not None:
        phase_journal.record(
            phase=residual_phase,
            event="end",
            started_monotonic=residual_started,
            success=True,
        )
    offgrid_residual = float(0.5 * np.mean(np.square(residual)))
    target_volume = float(boozer_surface.targetlabel)
    relative_volume_error = float(
        abs(certificate_surface.volume() - target_volume) / abs(target_volume)
    )
    iota_target_error = float(abs(float(refinement_result["iota"]) - iota_target))
    iota_refinement_delta = float(
        abs(float(refinement_result["iota"]) - float(coarse_result["iota"]))
    )
    relative_G_refinement_delta = float(
        abs(float(refinement_result["G"]) - float(coarse_result["G"]))
        / abs(float(coarse_result["G"]))
    )

    certificate = BoozerCertificate(
        offgrid_residual=offgrid_residual,
        relative_volume_error=relative_volume_error,
        iota_target_error=iota_target_error,
        iota_refinement_delta=iota_refinement_delta,
        relative_G_refinement_delta=relative_G_refinement_delta,
        refinement_nphi=len(refinement_surface.quadpoints_phi),
        refinement_ntheta=len(refinement_surface.quadpoints_theta),
        certificate_nphi=len(certificate_surface.quadpoints_phi),
        certificate_ntheta=len(certificate_surface.quadpoints_theta),
        fourier_mpol=refinement_surface.mpol,
        fourier_ntor=refinement_surface.ntor,
        spectral_order_certified=False,
    )
    require_boozer_certificate(
        certificate,
        CERTIFICATE_THRESHOLDS,
    )
    return refinement_boozer, certificate, refinement_report


def require_boozer_certificate(
    certificate: BoozerCertificate,
    thresholds: BoozerCertificateThresholds,
) -> None:
    """Reject a publication certificate when any numerical gate fails."""

    gates = {
        "off-grid Boozer residual": (
            certificate.offgrid_residual,
            thresholds.max_offgrid_residual,
        ),
        "relative volume error": (
            certificate.relative_volume_error,
            thresholds.max_relative_volume_error,
        ),
        "iota target error": (
            certificate.iota_target_error,
            thresholds.max_iota_error,
        ),
        "iota refinement delta": (
            certificate.iota_refinement_delta,
            thresholds.max_iota_refinement_delta,
        ),
        "relative G refinement delta": (
            certificate.relative_G_refinement_delta,
            thresholds.max_relative_G_refinement_delta,
        ),
    }
    invalid_thresholds = [
        name
        for name, (_, threshold) in gates.items()
        if not np.isfinite(threshold) or threshold < 0.0
    ]
    if invalid_thresholds:
        raise ValueError(
            "certificate thresholds must be finite and non-negative: "
            + ", ".join(invalid_thresholds)
        )
    failed_gates = [
        f"{name}={value:.6e} > {threshold:.6e}"
        for name, (value, threshold) in gates.items()
        if not np.isfinite(value) or value > threshold
    ]
    if failed_gates:
        raise BoozerSolveError("publication certificate failed: " + "; ".join(failed_gates))


def publish_boozer_artifact(
    boozer_surface,
    metadata: dict,
    *,
    output_directory: str | Path,
    TF_a: float,
):
    """Atomically publish one hash-bound surface and metadata generation."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    staging_path = Path(mkdtemp(prefix=".stage2-staging-", dir=output_path))
    try:
        staged_surface = staging_path / "optimized_boozer_surface.json"
        boozer_surface.save(staged_surface)
        surface_bytes = staged_surface.read_bytes()
        bound_metadata = {
            **metadata,
            "surface_sha256": sha256(surface_bytes).hexdigest(),
        }
        staged_metadata = staging_path / "metadata.json"
        metadata_bytes = json.dumps(
            bound_metadata,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        staged_metadata.write_bytes(metadata_bytes)
        publication_digest = sha256(surface_bytes + metadata_bytes).hexdigest()
        final_directory = output_path / (
            f"TF_a_{TF_a:.3f}_{publication_digest[:16]}"
        )
        if final_directory.exists():
            existing_surface = (
                final_directory / staged_surface.name
            ).read_bytes()
            existing_metadata = (
                final_directory / staged_metadata.name
            ).read_bytes()
            existing_digest = sha256(
                existing_surface + existing_metadata
            ).hexdigest()
            if existing_digest != publication_digest:
                raise FileExistsError(
                    f"publication digest prefix collision at {final_directory}"
                )
        else:
            staging_path.replace(final_directory)
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)
    return (
        final_directory / staged_surface.name,
        final_directory / staged_metadata.name,
        publication_digest,
    )


def polarity_output_directory(base_directory: str | Path, polarity: int) -> Path:
    suffix = "G_positive" if normalize_polarity(polarity) > 0 else "G_negative"
    return Path(base_directory) / suffix
