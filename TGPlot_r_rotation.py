# from pathlib import Path
# from simsopt._core import load, save
# import json

# from pathlib import Path
# import json
# import numpy as np

# from scipy.optimize import minimize

# from simsopt._core import load
# from simsopt.field import CurrentPenalty

# from simsopt.geo import (
#     BoozerSurface,
#     BoozerResidual,
#     Iotas,
#     NonQuasiSymmetricRatio,
#     SurfaceXYZTensorFourier,
#     Volume
# )
# from simsopt.objectives import QuadraticPenalty
# import os
# from simsopt.geo.curveplanarellipticalcylindrical import CurvePlanarEllipticalCylindrical

# TF_a_list_FCUG = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425]

# n_tf = 4
# rotations_before = [[] for _ in range(n_tf)]
# rotations_after = [[] for _ in range(n_tf)]

# for TF_a in TF_a_list_FCUG:
#     folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed/TF_a_{TF_a:.3f}"
#     bs = load(Path(folder) / "bs_opt.json")
#     surf = load(Path(folder) / "surf_opt.json")

#     for i, coil in enumerate(bs.coils[:n_tf]):
#         rotations_before[i].append(coil.curve.get_dofs()[0])

# from helper_functions import *

# for TF_a in TF_a_list_FCUG:
#     path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")
#     boozersurface = load(path / f"optimized_boozer_surface_{TF_a:.3f}.json")
#     bs = boozersurface.biotsavart
#     surf = boozersurface.surface

#     for i, coil in enumerate(bs.coils[:n_tf]):
#         rotations_after[i].append(coil.curve.get_dofs()[0])

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np

# fig, ax = plt.subplots(figsize=(7, 5.5))

# TF_a_all = TF_a_list_FCUG

# for i in range(n_tf):
#     print(f"coil {i}: before={rotations_before[i]}, after={rotations_after[i]}")

# colors = cm.tab10(np.linspace(0, 1, n_tf))

# for coil_idx in range(n_tf):
#     before = rotations_before[coil_idx]
#     after = rotations_after[coil_idx]
#     color = colors[coil_idx]

#     ax.scatter(
#         TF_a_all,
#         before,
#         color=color,
#         marker="o",
#         facecolors=color,
#         zorder=3,
#         label=f"TF coil {coil_idx+1} before",
#     )

#     ax.scatter(
#         TF_a_all,
#         after,
#         color=color,
#         marker="^",
#         facecolors="none",
#         edgecolors=color,
#         linewidths=1.5,
#         zorder=3,
#         label=f"TF coil {coil_idx+1} after",
#     )

#     for x, yb, ya in zip(TF_a_all, before, after):
#         ax.annotate(
#             "",
#             xy=(x, ya),
#             xycoords="data",
#             xytext=(x, yb),
#             textcoords="data",
#             arrowprops=dict(
#                 arrowstyle="->",
#                 color=color,
#                 lw=1.0,
#                 alpha=0.6,
#             ),
#         )

# ax.set_xlabel("TF_a")
# ax.set_ylabel("r_rotation (rad)")
# ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
# ax.set_title("TF coil r_rotation before vs. after optimization")

# plt.tight_layout()

# plt.savefig(
#     "/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs/major_radius_before_after.png",
#     dpi=150,
#     bbox_inches="tight",
# )

# ---------NEXT

# from pathlib import Path
# from simsopt._core import load, save
# import json

# from pathlib import Path
# import json
# import numpy as np

# from scipy.optimize import minimize

# from simsopt._core import load
# from simsopt.field import CurrentPenalty

# from simsopt.geo import (
#     BoozerSurface,
#     BoozerResidual,
#     Iotas,
#     NonQuasiSymmetricRatio,
#     SurfaceXYZTensorFourier,
#     Volume
# )
# from simsopt.objectives import QuadraticPenalty
# import os
# from simsopt.geo.curveplanarellipticalcylindrical import CurvePlanarEllipticalCylindrical

# TF_a_list_FCUG = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425]

# n_tf = 4
# rotations_before = [[] for _ in range(n_tf)]
# rotations_after = [[] for _ in range(n_tf)]

# for TF_a in TF_a_list_FCUG:
#     folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed/TF_a_{TF_a:.3f}"
#     bs = load(Path(folder) / "bs_opt.json")
#     surf = load(Path(folder) / "surf_opt.json")

#     for i, coil in enumerate(bs.coils[:n_tf]):
#         rotations_before[i].append(coil.curve.get_dofs()[3])

# from helper_functions import *

# for TF_a in TF_a_list_FCUG:
#     path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")
#     boozersurface = load(path / f"optimized_boozer_surface_{TF_a:.3f}.json")
#     bs = boozersurface.biotsavart
#     surf = boozersurface.surface

#     for i, coil in enumerate(bs.coils[:n_tf]):
#         rotations_after[i].append(coil.curve.get_dofs()[3])

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np

# fig, ax = plt.subplots(figsize=(7, 5.5))

# TF_a_all = TF_a_list_FCUG

# for i in range(n_tf):
#     print(f"coil {i}: before={rotations_before[i]}, after={rotations_after[i]}")

# colors = cm.tab10(np.linspace(0, 1, n_tf))

# for coil_idx in range(n_tf):
#     before = np.array(rotations_before[coil_idx])
#     after = np.array(rotations_after[coil_idx])
#     diff = after - before
#     color = colors[coil_idx]

#     print(f"coil {coil_idx}: diff (after-before) = {diff}")

#     ax.scatter(
#         TF_a_all,
#         diff,
#         color=color,
#         marker="o",
#         facecolors=color,
#         zorder=3,
#         label=f"TF coil {coil_idx+1}",
#     )

#     ax.plot(
#         TF_a_all,
#         diff,
#         color=color,
#         lw=1.0,
#         alpha=0.6,
#         zorder=2,
#     )

# ax.axhline(0, color="gray", lw=0.8, linestyle="--", zorder=1)
# ax.set_xlabel("TF_a")
# ax.set_ylabel("R0 change (after - before) [m]")
# ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
# ax.set_title("TF coil major radius (R0) shift after optimization")

# plt.tight_layout()

# plt.savefig(
#     "/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs/r_rotation_diff.png",
#     dpi=150,
#     bbox_inches="tight",
# )



from pathlib import Path
from simsopt._core import load, save
import json

from pathlib import Path
import json
import numpy as np

from scipy.optimize import minimize

from simsopt._core import load
from simsopt.field import CurrentPenalty

from simsopt.geo import (
    BoozerSurface,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume
)
from simsopt.objectives import QuadraticPenalty
import os
from simsopt.geo.curveplanarellipticalcylindrical import CurvePlanarEllipticalCylindrical

TF_a_list_FCUG = [0.300, 0.325, 0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.600, 0.625, 0.650, 0.825,0.850,0.875,0.900]

n_tf = 4
rotations_before = [[] for _ in range(n_tf)]
rotations_after = [[] for _ in range(n_tf)]

for TF_a in TF_a_list_FCUG:
    folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed1/TF_a_{TF_a:.3f}"
    bs = load(Path(folder) / "bs_initial.json")
    surf = load(Path(folder) / "surf_opt.json")

    for i, coil in enumerate(bs.coils[:n_tf]):
        rotations_before[i].append(coil.curve.get_dofs()[3])

for TF_a in TF_a_list_FCUG:
    folder = f"/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed1/TF_a_{TF_a:.3f}"
    bs = load(Path(folder) / "bs_opt.json")
    surf = load(Path(folder) / "surf_opt.json")

    for i, coil in enumerate(bs.coils[:n_tf]):
        rotations_after[i].append(coil.curve.get_dofs()[3])

from helper_functions import *


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5.5))

TF_a_all = TF_a_list_FCUG

for i in range(n_tf):
    print(f"coil {i}: before={rotations_before[i]}, after={rotations_after[i]}")

colors = cm.tab10(np.linspace(0, 1, n_tf))

for coil_idx in range(n_tf):
    before = np.array(rotations_before[coil_idx])
    after = np.array(rotations_after[coil_idx])
    diff = after - before
    color = colors[coil_idx]

    print(f"coil {coil_idx}: diff (after-before) = {diff}")

    ax.scatter(
        TF_a_all,
        diff,
        color=color,
        marker="o",
        facecolors=color,
        zorder=3,
        label=f"TF coil {coil_idx+1}",
    )

    ax.plot(
        TF_a_all,
        diff,
        color=color,
        lw=1.0,
        alpha=0.6,
        zorder=2,
    )

ax.axhline(0, color="gray", lw=0.8, linestyle="--", zorder=1)
ax.set_xlabel("TF_a")
ax.set_ylabel("R0 change (after - before) [m]")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
ax.set_title("TF coil major radius (R0) shift after optimization")

plt.tight_layout()

plt.savefig(
    "/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed1/r_rotation_diff2.png",
    dpi=150,
    bbox_inches="tight",
)