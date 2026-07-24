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


MU0 = 4e-7 * np.pi

base_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_VV_unfixed")

tf_vals = []
VV_vals= []
J_vals = []


import matplotlib.pyplot as plt


for folder in sorted(base_dir.glob("TF_a_*")):
    if not folder.is_dir():
        continue

    tf_a_tmp = float(folder.name.split("_")[2])
    VV_tmp = float(folder.name.split("_")[4])


    print(f"Processing TF_A = {tf_a_tmp}, {VV_tmp}")

    results_file = folder / "results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    J_cur = results["final_squared_flux"]
    tf_a_cur = results["TF_a"]
    VV_cur = VV_tmp

    tf_vals.append(tf_a_cur)
    VV_vals.append(VV_cur)
    J_vals.append(J_cur)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# sc = ax.scatter(tf_vals, VV_vals, J_vals, c=J_vals, cmap="viridis")

# ax.set_xlabel("TF_a")
# ax.set_ylabel("VV")
# ax.set_zlabel("J (final squared flux)")
# ax.set_title("J vs TF_a and VV")

# plt.colorbar(sc, ax=ax, label="J")

# plt.savefig(
#     "/burg-archive/home/tg2998/simsopt/examples/outputs/fixed/TF_VV_J_3D.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()


import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=tf_vals,
    y=VV_vals,
    z=J_vals,
    mode='markers',
    marker=dict(
        size=5,
        color=J_vals,
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    scene=dict(
        xaxis_title='TF_a',
        yaxis_title='VV',
        zaxis_title='J'
    ),
    title="Interactive 3D: J vs TF_a and VV"
)

# SAVE interactive file
fig.write_html(
    "/burg-archive/home/tg2998/simsopt/examples/outputs/unfixed/TF_VV_J_3D.html"
)

fig.show()