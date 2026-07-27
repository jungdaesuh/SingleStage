"""Immutable identity for a resumable single-stage optimization run."""

from dataclasses import asdict, dataclass
import os
import shutil
from typing import Any, Mapping


@dataclass(frozen=True)
class RunConfiguration:
    schema_version: int
    init_dir: str
    method: str
    mpol: int
    ntor: int
    field_polarity: int
    iota_target: float
    f_b_threshold: float
    f_cp_threshold: float
    iota_threshold: float
    iota_penalty_weight: float
    iota_scale: float
    volume_target: float
    maxiter: int
    outer_step_radius: float
    boozer_constraint_weight: float
    penalty_weight: float
    current_pnorm_p: float
    current_scale: float
    gtol: float
    sparse: bool
    theta_tol: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches(self, artifact: Mapping[str, Any]) -> bool:
        return artifact.get("run_config") == self.to_dict()

    def require_match(self, artifact: Mapping[str, Any], artifact_path: str) -> None:
        if not self.matches(artifact):
            raise RuntimeError(
                f"Existing artifact at {artifact_path} was produced with a "
                "different run configuration; pass --new to replace it."
            )


def prepare_output_generation(output_dir: str, start_fresh: bool) -> None:
    """Remove the prior run generation when the operator explicitly requests it."""
    if start_fresh and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
