"""Single source of truth for Tian's physical dipole/TF coil layout."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoilLayout:
    """Physical counts and symmetry-root counts for one field periodicity."""

    nfp: int
    ntf_roots: int
    wp_ntor: int
    tf_physical_count: int = 16
    dipole_poloidal_rows: int = 11
    dipole_toroidal_positions: int = 32

    @property
    def dipole_physical_count(self) -> int:
        return self.dipole_poloidal_rows * self.dipole_toroidal_positions

    @property
    def total_physical_count(self) -> int:
        return self.tf_physical_count + self.dipole_physical_count

    @property
    def tf_root_count(self) -> int:
        return self.ntf_roots

    @property
    def dipole_root_count(self) -> int:
        return 2 * self.wp_ntor * self.dipole_poloidal_rows

    def validate(self) -> None:
        if self.nfp not in (2, 4, 8):
            raise ValueError(f"unsupported Nfp={self.nfp}; expected one of 2, 4, 8")
        if self.ntf_roots * self.nfp * 2 != self.tf_physical_count:
            raise ValueError("TF symmetry roots do not expand to 16 physical coils")
        if 2 * self.wp_ntor * self.nfp != self.dipole_toroidal_positions:
            raise ValueError("dipole symmetry roots do not expand to 32 toroidal positions")


def layout_for_nfp(nfp: int) -> CoilLayout:
    """Return the only supported symmetry-root policy for ``Nfp``."""

    if nfp not in (2, 4, 8):
        raise ValueError(f"unsupported Nfp={nfp}; expected one of 2, 4, 8")
    layout = CoilLayout(
        nfp=nfp,
        ntf_roots=8 // nfp,
        wp_ntor=16 // nfp,
    )
    layout.validate()
    return layout
