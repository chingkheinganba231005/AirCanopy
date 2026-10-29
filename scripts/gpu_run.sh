#!/usr/bin/env bash
# Run AirCanopy's graphics-card jobs on any Linux machine with an NVIDIA card: a rented one
# (RunPod, Vast.ai), or a Colab runtime opened with the Colab CLI's `colab console`. After each
# job it packs everything the jobs have written so far into aircanopy_gpu_results.zip, the zip
# that scripts/colab_unpack.py checks, so a run that stops early still leaves its results.
#
# Start it so that it keeps running after the browser is closed (docs/gpu_rental.md):
#
#   cd /workspace && curl -fsSLO https://raw.githubusercontent.com/chingkheinganba231005/AirCanopy/main/scripts/gpu_run.sh
#   nohup setsid bash gpu_run.sh > gpu_run.log 2>&1 < /dev/null &
#   tail -n 30 gpu_run.log      # to see how far it has got
#
# Without arguments it runs the jobs listed in scripts/gpu_jobs.txt of the checked-out
# repository, so the list can change without changing this script. Each argument is otherwise
# one job, a script and its arguments, run with the project's Python:
#
#   bash gpu_run.sh "benchmarks/wind_speed.py 128 400"
#
# Settings, through the environment:
#   AIRCANOPY_REF  branch or commit to run (default: main)
#   AIRCANOPY_DIR  where to put the repository (default: /workspace/aircanopy if /workspace
#                  exists, else ./aircanopy)
#   JOB_HOURS      longest a job may run before it is stopped (default: 4)
#   GITHUB_TOKEN   only if the repository is private
#   STOP_POD=1     on RunPod, stop the pod when done, so that the card stops costing money.
#                  Stopping erases the pod's container disk, so this is done only when
#                  /workspace is a separate volume, which keeps the results zip
set -uo pipefail

REPO="github.com/chingkheinganba231005/AirCanopy.git"
REF="${AIRCANOPY_REF:-main}"
if [[ -z "${AIRCANOPY_DIR:-}" ]]; then
  if [[ -d /workspace && -w /workspace ]]; then AIRCANOPY_DIR=/workspace/aircanopy; else AIRCANOPY_DIR="$PWD/aircanopy"; fi
fi
JOBS=("$@")

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
if [[ ${#JOBS[@]} -eq 0 ]]; then
  mapfile -t JOBS < <(grep -Ev '^[[:space:]]*(#|$)' scripts/gpu_jobs.txt)
fi
[[ ${#JOBS[@]} -gt 0 ]] || die "no jobs"

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
zip="$AIRCANOPY_DIR/aircanopy_gpu_results.zip"
failed=0
packed=0
n=0
for job in "${JOBS[@]}"; do
  n=$((n + 1))
  say "job $n of ${#JOBS[@]}: $job"
  read -ra words <<<"$job"
  if ! timeout "${JOB_HOURS:-4}h" .venv/bin/python "${words[@]}"; then
    failed=$((failed + 1))
    say "job failed or ran out of time: $job"
  fi
  if .venv/bin/python scripts/gpu_pack.py --since "$start" --out "$zip" >/dev/null; then
    packed=1
    say "results so far are in $zip"
  fi
done

if ((packed)); then
  say "done: ${#JOBS[@]} jobs, $failed failed. Download $zip"
else
  say "done: ${#JOBS[@]} jobs, $failed failed; nothing was packed"
fi

if [[ "${STOP_POD:-0}" == 1 && -n "${RUNPOD_POD_ID:-}" ]]; then
  if grep -qs " /workspace " /proc/mounts && [[ "$zip" == /workspace/* ]]; then
    say "stopping pod $RUNPOD_POD_ID"
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" ||
      say "could not stop the pod: stop it from the RunPod console"
  else
    say "not stopping the pod: /workspace is not a separate volume, and stopping would erase the results"
  fi
fi
exit $((failed > 0 || packed == 0))
