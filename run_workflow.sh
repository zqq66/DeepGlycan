#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash run_workflow.sh [HYDRA_OVERRIDES...]

Runs the full pipeline:
  1) mgf_preprocess.py
  2) Hyper-Spec clustering
  3) pep_search.py
  4) main_inference_cluster.py
  5) main_FDR.py

All arguments are forwarded as Hydra overrides to the Python steps so the same
config is reused end-to-end.

Example:
  bash run_workflow.sh in_dir=/data/raw out_dir=/data/filtered
EOF
  exit 0
fi

OVERRIDES=("$@")

PARSE_OUTPUT="$(python - "${OVERRIDES[@]}" <<'PY'
from omegaconf import OmegaConf
import os, sys, traceback

try:
    cfg = OmegaConf.load("configs/config.yaml")
    override_cfg = OmegaConf.from_dotlist(sys.argv[1:]) if len(sys.argv) > 1 else OmegaConf.create()
    cfg = OmegaConf.merge(cfg, override_cfg)

    def clean(path):
        if path is None:
            return ""
        return os.path.expanduser(str(path))

    in_dir = clean(cfg.get("in_dir", ""))
    out_dir = clean(cfg.get("out_dir", ""))
    if not out_dir:
        out_dir = in_dir
    mgf_dir = out_dir or in_dir
    work_dir = clean(cfg.get("working_dir", ""))
    if not work_dir:
        work_dir = mgf_dir

    print(in_dir)
    print(out_dir)
    print(mgf_dir)
    print(work_dir)
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY
)" || {
  echo "Failed to resolve paths from config/overrides. See error above." >&2
  exit 1
}

read -r IN_DIR OUT_DIR MGF_DIR WORK_DIR <<<"$PARSE_OUTPUT"

mapfile -t PARSED <<<"$PARSE_OUTPUT"
IN_DIR="${PARSED[0]-}"
OUT_DIR="${PARSED[1]-}"
MGF_DIR="${PARSED[2]-}"
WORK_DIR="${PARSED[3]-}"

if [[ -z "${MGF_DIR}" ]]; then
  echo "Could not resolve in_dir/out_dir. Parsed output:" >&2
  printf '%s\n' "$PARSE_OUTPUT" >&2
  echo "Set them via configs/config.yaml or CLI: in_dir=/path/to/mgf out_dir=/path/to/output" >&2
  exit 1
fi

echo "Hydra overrides: ${OVERRIDES[*]:-"<none>"}"
echo "Input dir : ${IN_DIR:-"<unset>"}"
echo "Output dir: ${OUT_DIR}"
echo "MGF dir   : ${MGF_DIR}"
echo "Work dir  : ${WORK_DIR}"

echo "Step 1/5: mgf_preprocess.py"
# python mgf_preprocess.py "${OVERRIDES[@]}"

echo "Step 2/5: Hyper-Spec clustering"
# (
#   cd Hyper-Spec
#   python src/main.py "$MGF_DIR" "$MGF_DIR/cluster.csv" --cpu_core_preprocess=24 --cluster_alg dbscan --use_gpu_cluster --cluster_charges 2 3 4 5 --eps=0.2
# )

echo "Step 3/5: pep_search.py"
# python pep_search.py "${OVERRIDES[@]}"

echo "Step 4/5: main_inference_cluster.py"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_inference_cluster.py "${OVERRIDES[@]}"

echo "Step 5/5: main_FDR.py"
python main_FDR.py "${OVERRIDES[@]}"
