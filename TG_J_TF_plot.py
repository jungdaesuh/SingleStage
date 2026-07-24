from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

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

# Original base_dir for the VV_unfixed runs
base_dir = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_unfixed_True")

tf_vals = []
J_vals = []

for folder in sorted(base_dir.glob("TF_a_*")):
    if not folder.is_dir():
        continue

    # Using the parsing logic from your original script to handle the VV folders
    tf_a_tmp = float(folder.name.split("_")[2])

    print(f"Processing TF_A = {tf_a_tmp}")

    results_file = folder / "results.json"

    with open(results_file, "r") as f:
        results = json.load(f)

    J_cur = results["final_squared_flux"]
    tf_a_cur = results["TF_a"]

    tf_vals.append(tf_a_cur)
    J_vals.append(J_cur)

# ---------------------------------------------------------
# Sorting and Plotting (Using your updated template format)
# ---------------------------------------------------------

# Sort by TF_a
tf_vals, J_vals = zip(*sorted(
    zip(tf_vals, J_vals),
    key=lambda x: x[0]
))

# Hardcode colors to green since able_to_initialise is not present here
colors = ["green" for _ in tf_vals]

plt.figure()
plt.scatter(tf_vals, J_vals, c=colors, marker="o")
plt.plot(tf_vals, J_vals, linestyle="--", alpha=0.5)  # line

plt.xlabel("TF_a")
plt.ylabel("Final squared flux (J)")
plt.title("J vs TF_a")
plt.grid(True)

save_path = "/burg-archive/home/tg2998/simsopt/examples/outputs/unfixed_True/TF_vs_J.png"
plt.savefig(
    save_path,
    dpi=300, 
    bbox_inches="tight"
)

print(f"Plot successfully saved to: {save_path}")

plt.show()