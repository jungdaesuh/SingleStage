from pathlib import Path
from simsopt._core import load, save
import json


TF_a_list_FCUG = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425]
J_before = []
J_after = []

for TF_a in TF_a_list_FCUG:
    folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed/TF_a_{TF_a:.3f}"
    results_file = Path(folder) / "results.json"
    with open(results_file, "r") as f:
        results = json.load(f)
    J_cur = results["final_squared_flux"]
    J_before.append(J_cur)

from helper_functions import *

for TF_a in TF_a_list_FCUG:
    path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")
    boozersurface = load(path / f"optimized_boozer_surface_{TF_a:.3f}.json")
    print(boozersurface.boozer_type)  # 'ls' or 'exact'
    bs = boozersurface.biotsavart
    surf = boozersurface.surface

    #bs = load(path / f"optimized_bs_{TF_a:.3f}.json")
    #surf = load(path / f"optimized_surface_{TF_a:.3f}.json")
    Jf = SquaredFlux(surf, bs, definition="local")
    J_after.append(Jf.J())


# TF_a_list_FCFG = [0.675, 0.700]
# for TF_a in TF_a_list_FCFG:
#     folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed_True/TF_a_{TF_a:.3f}"
#     results_file = Path(folder) / "results.json"
#     with open(results_file, "r") as f:
#         results = json.load(f)
#     J_cur = results["final_squared_flux"]
#     J_before.append(J_cur)

# for TF_a in TF_a_list_FCFG:
#     path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")
#     bs = load(path / f"optimized_bs_{TF_a:.3f}.json")
#     surf = load(path / f"optimized_surface_{TF_a:.3f}.json")
#     Jf = SquaredFlux(surf, bs, definition="local")
#     J_after.append(Jf.J())





# import matplotlib.pyplot as plt 
# # --- Plot ---
# fig, ax = plt.subplots(figsize=(6, 5))

# TF_a_all = TF_a_list_FCUG # + TF_a_list_FCFG


# for TF_a, jb, ja in zip(TF_a_all, J_before, J_after):
#     ax.scatter(TF_a, jb, color="tab:red", zorder=3, label="before" if TF_a == TF_a_list_FCUG[0] else None)
#     ax.scatter(TF_a, ja, color="tab:blue", zorder=3, label="after" if TF_a == TF_a_list_FCUG[0] else None)
#     ax.annotate(
#         "",
#         xy=(TF_a, ja), xycoords="data",
#         xytext=(TF_a, jb), textcoords="data",
#         arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
#     )

# ax.set_xlabel("TF_a")
# ax.set_ylabel("Squared flux J")
# #ax.set_yscale("log")  # squared-flux values often span orders of magnitude; drop this if not needed
# ax.legend()
# ax.set_title("Squared flux before vs. after single-stage optimization")
# plt.tight_layout()
# plt.savefig("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs/J_before_after_FCUG_FCFG.png", dpi=150)
