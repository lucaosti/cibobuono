#!/usr/bin/env bash
# setup_environment.sh — Bootstrap a fresh CiboBuono pipeline host.
#
# Installs system tools, creates a Python venv, installs dependencies (with
# CUDA-enabled llama-cpp-python when an NVIDIA GPU is present), downloads
# models that cannot live in git, and runs a quick verification suite.
#
# Usage (from repo root):
#   bash scripts/setup_environment.sh              # full setup
#   bash scripts/setup_environment.sh --skip-models  # deps + venv only
#   bash scripts/setup_environment.sh --verify-only  # re-run checks
#
# Environment overrides:
#   CIBOBUONO_PYTHON     Python executable (default: python3.11 || python3.10 || python3)
#   CIBOBUONO_VENV_DIR   venv path (default: $REPO/.venv)
#   CIBOBUONO_LLM_TIER   passed to setup_models --tier

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CIBOBUONO_VENV_DIR:-$REPO/.venv}"
SKIP_MODELS=0
VERIFY_ONLY=0
FORCE_LLM=0

for arg in "$@"; do
  case "$arg" in
    --skip-models)   SKIP_MODELS=1 ;;
    --verify-only)   VERIFY_ONLY=1 ;;
    --force-llm)     FORCE_LLM=1 ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }

# ── Python ───────────────────────────────────────────────────────────────────

pick_python() {
  if [[ -n "${CIBOBUONO_PYTHON:-}" ]]; then
    echo "$CIBOBUONO_PYTHON"
    return
  fi
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" &>/dev/null; then
      ver=$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
      major=${ver%%.*}
      minor=${ver#*.}
      if (( major > 3 || (major == 3 && minor >= 10) )); then
        echo "$cand"
        return
      fi
    fi
  done
  echo ""
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  echo "ERROR: Python ≥3.10 required." >&2
  exit 1
fi
log "Python: $($PY --version) ($PY)"

# ── System packages (best-effort) ────────────────────────────────────────────

install_system_deps() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    log "Non-Linux host — install ffmpeg manually if missing (brew install ffmpeg)."
    command -v ffmpeg &>/dev/null || warn "ffmpeg not found on PATH"
    return
  fi
  if command -v apt-get &>/dev/null && [[ "$(id -u)" -eq 0 || -n "${SUDO:-}" ]]; then
    SUDO_CMD=""
    [[ "$(id -u)" -ne 0 ]] && SUDO_CMD="sudo"
    log "Installing system packages (ffmpeg, git, build tools)…"
    $SUDO_CMD apt-get update -qq
    $SUDO_CMD apt-get install -y -qq \
      ffmpeg git curl ca-certificates \
      python3-venv python3-dev build-essential cmake pkg-config \
      2>/dev/null || warn "Some apt packages failed — continue if ffmpeg is present"
  else
    command -v ffmpeg &>/dev/null || warn "Install ffmpeg manually (apt/brew)."
  fi
}

has_cuda() {
  # nvidia-smi is the cleanest signal but is often absent even when the driver
  # and toolkit are installed (headless boxes), so probe several sources.
  command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null && return 0
  [[ -e /dev/nvidia0 ]] && return 0
  [[ -f /proc/driver/nvidia/version ]] && return 0
  command -v lspci &>/dev/null && lspci 2>/dev/null | grep -qi 'nvidia' && return 0
  return 1
}

has_nvcc() { command -v nvcc &>/dev/null; }

# ── Venv + pip ───────────────────────────────────────────────────────────────

setup_venv() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    log "Creating venv at $VENV"
    "$PY" -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  log "Upgrading pip…"
  python -m pip install -U pip wheel setuptools -q

  log "Installing Python dependencies…"
  python -m pip install -r "$REPO/requirements.txt" -q
  python -m pip install huggingface_hub -q

  # PyTorch: the default PyPI wheel already bundles CUDA on Linux, so a plain
  # install gives GPU-enabled torch (used by GLiNER/transformers).
  python -c "import torch" 2>/dev/null || python -m pip install torch -q || true

  if has_cuda; then
    # Is llama-cpp-python already a CUDA build? (requirements.txt pulls a
    # CPU-only wheel, so we usually need to rebuild it here.)
    local gpu_ok
    gpu_ok="$(python -c 'from llama_cpp import llama_cpp as c; print(c.llama_supports_gpu_offload())' 2>/dev/null || echo False)"
    if [[ "$gpu_ok" == "True" ]]; then
      log "llama-cpp-python already supports GPU offload."
    elif has_nvcc; then
      log "NVIDIA GPU + nvcc detected — building llama-cpp-python with CUDA (this can take a few minutes)…"
      CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install llama-cpp-python \
        --no-cache-dir --force-reinstall --no-binary llama-cpp-python -q \
        && log "CUDA llama-cpp built." \
        || warn "CUDA llama-cpp build failed; LLM will run on CPU. Check CUDA toolkit."
    else
      warn "NVIDIA GPU present but nvcc missing — install 'nvidia-cuda-toolkit' to enable GPU LLM. LLM runs on CPU for now."
    fi
  else
    log "No NVIDIA GPU — CPU/MPS llama-cpp wheel is fine."
  fi
}

run_verify() {
  cd "$REPO"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  log "Hardware profile:"
  python -m scripts.hardware | head -40
  log "Running tests…"
  python -m pytest tests/ -q --tb=no -p no:cacheprovider
  log "Validating data JSON…"
  python -m scripts.validate_data
  if [[ "$SKIP_MODELS" -eq 0 ]]; then
    log "Verifying model assets…"
    python -m scripts.setup_models --verify || warn "Some models missing — run without --skip-models"
  fi
  log "Pipeline status:"
  python -m scripts.run_pipeline --status || true
}

# ── Main ─────────────────────────────────────────────────────────────────────

cd "$REPO"
mkdir -p "$REPO/cache" "$REPO/logs" "$REPO/models" "$REPO/data"

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  run_verify
  exit 0
fi

install_system_deps
setup_venv

if [[ "$SKIP_MODELS" -eq 0 ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  LLM_ARGS=()
  [[ -n "${CIBOBUONO_LLM_TIER:-}" ]] && LLM_ARGS+=(--tier "$CIBOBUONO_LLM_TIER")
  [[ "$FORCE_LLM" -eq 1 ]] && LLM_ARGS+=(--force-llm)
  log "Downloading models (Whisper + NER + LLM)…"
  python -m scripts.setup_models "${LLM_ARGS[@]}"
fi

run_verify
log "Setup complete. Activate with: source $VENV/bin/activate"
log "Run pipeline:  python -m scripts.run_pipeline --skip-push --max-videos 1 --no-dashboard"
