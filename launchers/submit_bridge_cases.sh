#!/usr/bin/env bash
# Preflight cases.tsv, then submit one SLURM array task per row.
#
#   ./submit_bridge_cases.sh              # preflight + sbatch
#   ./submit_bridge_cases.sh --dry-run    # print the driver commands, submit nothing
#   ./submit_bridge_cases.sh --local      # run rows sequentially on this machine (no SLURM)
#
# Environment (same names as stage2_to_single_stage.sh):
#   CASES        cases file                (default: ./cases.tsv next to this script)
#   PYTHON_BIN   fork venv python          (CurrentPenalty/ground are fork-only)
#   DRIVER_PATH  stage2_to_single_stage.py
#   STAGE2_ROOT  seed root containing TF_a_X.XXX/ dirs  (default: Ginsburg path)
#   OUTPUT_ROOT  output root                            (default: Ginsburg path)
#   MAX_ARRAY    maximum rows accepted                  (default: 50)
#   MAX_CONCURRENT maximum simultaneous Slurm tasks     (default: 3)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
CASES=${CASES:-${here}/cases.tsv}
PYTHON_BIN=${PYTHON_BIN:-/burg-archive/home/tg2998/simsopt/venv/bin/python}
DRIVER_PATH=${DRIVER_PATH:-/burg-archive/home/tg2998/simsopt/examples/dipoles/stage2_to_single_stage.py}
MAX_ARRAY=${MAX_ARRAY:-50}
MAX_CONCURRENT=${MAX_CONCURRENT:-3}
DEFAULT_STAGE2_ROOT=/burg-archive/home/tg2998/simsopt/examples/outputs/TG_warmstarter_stage2results_fixed
DEFAULT_OUTPUT_ROOT=/burg-archive/home/tg2998/simsopt/examples/outputs/TG_single_stage_outputs
STAGE2_ROOT=${STAGE2_ROOT:-${DEFAULT_STAGE2_ROOT}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}

mode=submit
case "${1:-}" in
  --dry-run) mode=dry ;;
  --local)   mode=local ;;
  "") ;;
  *) echo "unknown option: $1 (expected --dry-run or --local)"; exit 2 ;;
esac

[[ -f "${CASES}" ]] || { echo "cases file not found: ${CASES}"; exit 1; }
[[ "${MAX_ARRAY}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "MAX_ARRAY must be a positive integer, got '${MAX_ARRAY}'"; exit 1; }
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "MAX_CONCURRENT must be a positive integer, got '${MAX_CONCURRENT}'"; exit 1; }

mapfile -t rows < <(
  awk 'NF && $0 !~ /^[[:space:]]*#/' "${CASES}"
)
n=${#rows[@]}
(( n > 0 )) || { echo "no case rows in ${CASES}"; exit 1; }
if (( n > MAX_ARRAY )); then
  echo "refusing to queue ${n} cases (cap MAX_ARRAY=${MAX_ARRAY}); raise it explicitly if intended"
  exit 1
fi

is_number() {
  [[ "$1" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] \
    && awk -v value="$1" \
      'BEGIN { numeric=value+0; exit !(numeric >= -1.7e308 && numeric <= 1.7e308) }'
}

is_positive_number() {
  is_number "$1" && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

parse_row() {
  local row=$1
  local -a fields
  IFS=$'\t' read -r -a fields <<<"${row}"
  (( ${#fields[@]} == 9 )) \
    || { echo "row ${i}: expected 9 tab-separated columns, got ${#fields[@]}"; return 1; }
  TF_A=${fields[0]}
  POLARITY=${fields[1]}
  IOTA_TARGET=${fields[2]}
  CURRENT_LIMIT=${fields[3]}
  MPOL=${fields[4]}
  NTOR=${fields[5]}
  MAXITER=${fields[6]}
  STEP_RADIUS=${fields[7]}
  VOLUME_TARGET=${fields[8]}
}

# Preflight: every row parses, and its seed directory has all three files.
i=0
for row in "${rows[@]}"; do
  i=$((i + 1))
  parse_row "${row}" || exit 1
  for field in TF_A CURRENT_LIMIT STEP_RADIUS; do
    is_positive_number "${!field}" \
      || { echo "row ${i}: ${field}='${!field}' must be finite and positive"; exit 1; }
  done
  is_number "${IOTA_TARGET}" \
    || { echo "row ${i}: IOTA_TARGET='${IOTA_TARGET}' must be finite"; exit 1; }
  for field in MPOL NTOR MAXITER; do
    [[ "${!field}" =~ ^[1-9][0-9]*$ ]] \
      || { echo "row ${i}: ${field}='${!field}' must be a positive integer"; exit 1; }
  done
  if [[ "${VOLUME_TARGET}" != "-" ]]; then
    is_positive_number "${VOLUME_TARGET}" \
      || { echo "row ${i}: VOLUME_TARGET='${VOLUME_TARGET}' must be '-' or finite and positive"; exit 1; }
  fi
  [[ "${POLARITY}" == "1" || "${POLARITY}" == "-1" ]] || { echo "row ${i}: polarity must be 1 or -1"; exit 1; }
  seed_dir=$(printf '%s/TF_a_%.3f' "${STAGE2_ROOT}" "${TF_A}")
  for f in bs_opt.json surf_opt.json results.json; do
    [[ -f "${seed_dir}/${f}" ]] || { echo "row ${i}: missing ${seed_dir}/${f}"; exit 1; }
  done
done
echo "preflight OK: ${n} case(s) in ${CASES}"

build_args() {
  local row=$1
  parse_row "${row}"
  args=("${TF_A}" --field-polarity "${POLARITY}" --iota-target "${IOTA_TARGET}"
        --current-limit-amperes "${CURRENT_LIMIT}" --mpol "${MPOL}" --ntor "${NTOR}"
        --maxiter "${MAXITER}" --outer-step-radius "${STEP_RADIUS}")
  [[ "${VOLUME_TARGET}" != "-" ]] && args+=(--volume-target "${VOLUME_TARGET}")
  args+=(--stage2-root "${STAGE2_ROOT}" --output-root "${OUTPUT_ROOT}")
}

case "${mode}" in
  dry)
    i=0
    for row in "${rows[@]}"; do
      i=$((i + 1)); build_args "${row}"
      echo "case ${i}: ${PYTHON_BIN} -u ${DRIVER_PATH} ${args[*]}"
    done
    ;;
  local)
    i=0
    for row in "${rows[@]}"; do
      i=$((i + 1)); build_args "${row}"
      echo "=== local case ${i}/${n}: ${args[*]}"
      HWLOC_COMPONENTS=-gl "${PYTHON_BIN}" -u "${DRIVER_PATH}" "${args[@]}"
    done
    ;;
  submit)
    mkdir -p logs
    submission_cases=$(mktemp "$(pwd -P)/logs/stage2_bridge_cases.XXXXXX.tsv")
    printf '%s\n' "${rows[@]}" > "${submission_cases}"
    CASES=${submission_cases}
    export CASES PYTHON_BIN DRIVER_PATH STAGE2_ROOT OUTPUT_ROOT
    echo "validated case snapshot: ${CASES}"
    sbatch --array="1-${n}%${MAX_CONCURRENT}" --export=ALL "${here}/run_stage2_bridge.sbatch"
    ;;
esac
