from pathlib import Path
from simsopt._core import load, save

path = Path("/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs")

print("Loading BoozerSurface...")
bsurf = load(path / "optimized_boozer_surface_0.675.json")

print("Saving BiotSavart...")
save(bsurf.biotsavart, path / "optimized_bs_0.675.json")

print("Saving surface...")
save(bsurf.surface, path / "optimized_surface_0.675.json")



# from helper_functions import *

# print("Loading BiotSavart...")
# bs = load(path / "optimized_bs.json")

# print("Loading Surface...")
# surf = load(path / "optimized_surface.json")
# Jf = SquaredFlux(surf, bs, definition="local")

# print(Jf.J())
