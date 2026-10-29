#!/usr/bin/env bash
# Run AirCanopy's graphics-card jobs on any Linux machine with an NVIDIA card: a rented one
# (RunPod, Vast.ai), or a Colab runtime opened with the Colab CLI's `colab console`. It runs
# the same jobs as colab/wind_gpu.ipynb unless others are given, then packs what they wrote
# into aircanopy_gpu_results.zip for scripts/colab_unpack.py, as the notebook does.
#
# Start it so that it keeps running when the connection drops or the laptop sleeps
# (docs/gpu_rental.md):
#
#   curl -fsSLO https://raw.githubusercontent.com/chingkheinganba231005/AirCanopy/main/scripts/gpu_run.sh
#   nohup bash gpu_run.sh > gpu_run.log 2>&1 &
#   tail -f gpu_run.log      # to watch; Ctrl-C stops the watching, not the run
#
# Each argument is one job, a script and its arguments, run with the project's Python:
#
#   bash gpu_run.sh "benchmarks/wind_speed.py 128 400"
#
# Settings, through the environment:
#   AIRCANOPY_REF  branch or commit to run (default: main)
#   AIRCANOPY_DIR  where to put the repository (default: /workspace/aircanopy if /workspace
#                  exists, which RunPod keeps when a pod is stopped; else ./aircanopy)
#   GITHUB_TOKEN   only if the repository is private
#   STOP_POD=1     on RunPod, stop the pod when done: the card is released and billing for
#                  it ends; /workspace and the results zip stay, and the disk is still billed
set -uo pipefail

REPO="github.com/chingkheinganba231005/AirCanopy.git"
REF="${AIRCANOPY_REF:-main}"
if [[ -z "${AIRCANOPY_DIR:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then AIRCANOPY_DIR=/workspace/aircanopy; else AIRCANOPY_DIR="$PWD/aircanopy"; fi
fi
JOBS=("$@")
if [[ ${#JOBS[@]} -eq 0 ]]; then # the notebook's jobs
  JOBS=("benchmarks/wind_speed.py 128 400" "benchmarks/wind_speed.py 256 200"
    "benchmarks/wind_tgv1600.py 128 256" "benchmarks/wind_tgv1600.py 128 256 --lattice D3Q19")
fi

say() { printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() {
  say "stopped: $*"
  exit 1
}

say "graphics card"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || die "no NVIDIA card or driver"

say "repository $REPO at $REF in $AIRCANOPY_DIR"
url="https://${GITHUB_TOKEN:+$GITHUB_TOKEN@}$REPO"
if [[ ! -d "$AIRCANOPY_DIR/.git" ]]; then
  git clone --quiet --depth 1 "$url" "$AIRCANOPY_DIR" || die "clone failed"
fi
cd "$AIRCANOPY_DIR" || die "no $AIRCANOPY_DIR"
if ! { git fetch --quiet --depth 1 origin "$REF" && git checkout --quiet --detach FETCH_HEAD; }; then
  die "cannot check out $REF"
fi
git log -1 --format='%H %s'

say "Python environment"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  python3 -m pip install --quiet --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh || die "cannot install uv"
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked --quiet || die "uv sync failed"
# The same JAX as pyproject.toml, with its CUDA 12 libraries, no newer than the project's
# package cut-off date (D-017).
read -r jax cutoff < <(.venv/bin/python -c 'import tomllib
p = tomllib.load(open("pyproject.toml", "rb"))
jax = next(d for d in p["project"]["dependencies"] if d.startswith("jax=="))
print(jax.split("==")[1], p["tool"]["uv"]["exclude-newer"])')
uv pip install --quiet --python .venv "jax[cuda12]==$jax" --exclude-newer "$cutoff" || die "cannot install JAX for CUDA"
.venv/bin/python -c 'import jax; d = jax.devices(); print(jax.__version__, d); assert d[0].platform == "gpu"' ||
  die "JAX does not see the graphics card"

start=$(date +%s)
failed=0
for job in "${JOBS[@]}"; do
  say "job: $job"
  read -ra words <<<"$job"
  if ! .venv/bin/python "${words[@]}"; then
    failed=$((failed + 1))
    say "job failed: $job"
  fi
done

say "packing the results"
zip="$AIRCANOPY_DIR/aircanopy_gpu_results.zip"
if .venv/bin/python scripts/gpu_pack.py --since "$start" --out "$zip"; then
  packed=1
  say "done: ${#JOBS[@]} jobs, $failed failed. Download $zip"
else
  packed=0
  say "done: ${#JOBS[@]} jobs, $failed failed; nothing was packed"
fi

if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]]; then
  say "stopping pod $RUNPOD_POD_ID"
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" ||
    say "could not stop the pod: stop it from the RunPod console"
fi
exit $((failed > 0 || packed == 0))
