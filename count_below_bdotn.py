#!/usr/bin/env python3
"""
Count stage-2 run directories with average B·n below a threshold.
Usage: python count_below_bdotn.py [scan_dir] [threshold]
  scan_dir: parent dir of run subdirs (default: ../outputs/stage_2_LHS_4D_scan)
  threshold: max avg_Bnormal to count (default: 5e-3)
"""
import os
import sys
import json
import glob

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_scan = os.path.join(script_dir, "../outputs/stage_2_LHS_3D_scan_no_sparsity")
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else default_scan
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-3
    max_wp_current_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 2e5 # 200kA

    run_dirs = [d for d in glob.glob(os.path.join(scan_dir, "*")) if os.path.isdir(d)]
    below_dirs = []
    total_with_results = 0

    for run_dir in run_dirs:
        results_path = os.path.join(run_dir, "results.json")
        if not os.path.isfile(results_path):
            continue
        total_with_results += 1
        try:
            with open(results_path) as f:
                data = json.load(f)
            avg_bn = data.get("avg_Bnormal")
            max_wp = data.get("max_wp_current")
            ntoroidal = data.get("ntoroidal")
            npoloidal = data.get("npoloidal")
            poloidal_radius = data.get("poloidal_radius")
            Rtor_inboard = data.get("toroidal_radius_inboard")
            if avg_bn is not None and avg_bn < threshold and max_wp is not None and max_wp < max_wp_current_threshold:
                below_dirs.append((os.path.basename(run_dir), avg_bn, max_wp, ntoroidal, npoloidal, poloidal_radius, Rtor_inboard))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skip {os.path.basename(run_dir)}: {e}", file=sys.stderr)

    print(f"Scan dir: {scan_dir}")
    print(f"Threshold: avg_Bnormal < {threshold}")
    print(f"Threshold: max_wp_current < {max_wp_current_threshold}")
    print(f"Runs with results: {total_with_results}")
    print(f"Runs below thresholds: {len(below_dirs)}")
    if below_dirs:
        print("\nDirectories below threshold:")
        for name, avg_bn, max_wp, ntoroidal, npoloidal, poloidal_radius, Rtor_inboard in sorted(below_dirs, key=lambda x: x[1]):
            num = int(name.split("_")[0])
            if max_wp is not None:
                print(f"  {num}  (avg_Bnormal = {avg_bn:.2e}, max_wp_current = {max_wp:.2e} A, ntor = {ntoroidal}, npol = {npoloidal}, pol_rad = {poloidal_radius:.4f}, tor_rad_inboard = {Rtor_inboard:.4f})")
            else:
                print(f"  {num}  (avg_Bnormal = {avg_bn:.2e}, max_wp_current = N/A, ntor = {ntoroidal}, npol = {npoloidal}, pol_rad = {poloidal_radius:.4f}, tor_rad_inboard = {Rtor_inboard:.4f})")

if __name__ == "__main__":
    main()
