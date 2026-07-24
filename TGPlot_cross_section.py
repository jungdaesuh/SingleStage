from pathlib import Path
from simsopt._core import load
import json
import matplotlib.pyplot as plt
import numpy as np

TF_a_list_FCUG = [0.850]

save_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")

phis = [0.0, 0.25, 0.50, 0.75]

for TF_a in TF_a_list_FCUG:

    # ==========================
    # Stage II result
    # ==========================
    folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed/TF_a_{TF_a:.3f}"

    surf_before = load(Path(folder) / "surf_opt.json")


    # ==========================
    # Single-stage result
    # ==========================
    path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")

    boozersurface = load(path / f"optimized_boozer_surface_{TF_a:.3f}.json")
    surf_after = boozersurface.surface


    # ==========================
    # 4 cross sections
    # ==========================
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, phi in zip(axes, phis):

        cs_before = surf_before.cross_section(phi=phi)
        R_before = np.sqrt(cs_before[:,0]**2 + cs_before[:,1]**2)
        Z_before = cs_before[:,2]

        cs_after = surf_after.cross_section(phi=phi)
        R_after = np.sqrt(cs_after[:,0]**2 + cs_after[:,1]**2)
        Z_after = cs_after[:,2]

        ax.plot(R_before, Z_before, label="Stage II", linewidth=2)
        ax.plot(R_after, Z_after, label="Single stage", linewidth=2)

        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax.set_aspect("equal")
        ax.set_title(rf"$\phi={phi}$")

        ax.legend()

    fig.suptitle(f"Surface comparison TF_a={TF_a:.3f}", fontsize=16)
    plt.tight_layout()

    # save
    outfile = save_dir / f"surface_comparison_TF_a_{TF_a:.3f}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {outfile}")