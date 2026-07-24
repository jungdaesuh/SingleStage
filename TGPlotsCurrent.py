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

base_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_weights")

CT_vals = []
CW_vals= []
J_vals = []


import matplotlib.pyplot as plt


for folder in sorted(base_dir.glob("TF_a_*")):
    if not folder.is_dir():
        continue


    results_file = folder / "results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    J_cur = results["final_squared_flux"]
    CT_cur = results["current_threshold"]
    CW_cur = results["current_weight"]


    CT_vals.append(CT_cur)
    CW_vals.append(CW_cur)
    J_vals.append(J_cur)


import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=CT_vals,
    y=CW_vals,
    z=J_vals,
    mode='markers',
    marker=dict(
        size=5,
        color=J_vals,
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='J')
    )
)])

fig.update_layout(
    scene=dict(
        xaxis_title='Current Threshold (CT)',
        yaxis_title='Current Weight (CW)',
        xaxis_type='log',
        yaxis_type='log',
        zaxis_title='J'
    ),
    title="Interactive 3D: J vs Current Threshold and Current Weight"
)

# SAVE interactive file
fig.write_html(
    "/burg-archive/home/tg2998/simsopt/examples/outputs/unfixed/CT_CW_J_3D.html"
)

