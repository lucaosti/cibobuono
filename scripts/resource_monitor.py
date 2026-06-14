"""
resource_monitor.py — Live, runtime hardware governance.

Where :mod:`scripts.hardware` answers *"how much hardware exists?"* (a static,
cached snapshot of total capacity taken once at startup), this module answers
*"how much is free right now?"* and turns that into safe, adaptive decisions so
the pipeline never saturates the OS.

Responsibilities:
- :func:`snapshot` — instantaneous RAM / swap / CPU-load / GPU-VRAM reading.
- :func:`under_pressure` — is the system currently stressed?
- :func:`wait_until_calm` — block (interruptibly) until pressure clears or a
  timeout elapses.  Used as back-pressure between heavy work units.
- :func:`pick_llm_model` — choose the largest GGUF that *currently* fits free
  memory (VRAM when offloading to CUDA, otherwise system RAM), waiting briefly
  for headroom first and downgrading the tier if it never arrives
  ("wait, then downgrade").
- :func:`apply_friendly_priority` — drop process priority so the pipeline
  yields to interactive work on shared machines.

Everything degrades gracefully when ``psutil`` or ``nvidia-smi`` are missing:
callers always get a usable (if conservative) answer.

All thresholds are overridable via ``CIBOBUONO_*`` environment variables so the
same code adapts from a Raspberry Pi to a dedicated CUDA box without edits.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("resource_monitor")


# ---------------------------------------------------------------------------
# Tunable thresholds (env-overridable)
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %.2f", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Reserve this much RAM for the OS no matter what (GiB).
MIN_FREE_RAM_GB = _env_float("CIBOBUONO_MIN_FREE_RAM_GB", 2.0)
# A model of size S needs S * factor free before we dare to load it.
RAM_HEADROOM_FACTOR = _env_float("CIBOBUONO_RAM_HEADROOM_FACTOR", 1.35)
# Same idea for GPU VRAM (CUDA full offload).
MIN_FREE_VRAM_GB = _env_float("CIBOBUONO_MIN_FREE_VRAM_GB", 0.6)
VRAM_HEADROOM_FACTOR = _env_float("CIBOBUONO_VRAM_HEADROOM_FACTOR", 1.15)

# "Under pressure" trip points.
PRESSURE_RAM_PERCENT = _env_float("CIBOBUONO_PRESSURE_RAM_PERCENT", 88.0)
PRESSURE_SWAP_PERCENT = _env_float("CIBOBUONO_PRESSURE_SWAP_PERCENT", 25.0)
PRESSURE_LOAD_PER_CORE = _env_float("CIBOBUONO_PRESSURE_LOAD_PER_CORE", 2.5)
PRESSURE_VRAM_PERCENT = _env_float("CIBOBUONO_PRESSURE_VRAM_PERCENT", 92.0)

# "wait, then downgrade" timing.
HEADROOM_WAIT_SECONDS = _env_float("CIBOBUONO_HEADROOM_WAIT_SECONDS", 90.0)
PRESSURE_WAIT_SECONDS = _env_float("CIBOBUONO_PRESSURE_WAIT_SECONDS", 120.0)
POLL_SECONDS = max(0.5, _env_float("CIBOBUONO_POLL_SECONDS", 3.0))

# Only pin the model in RAM (mlock) when this much would remain free afterwards.
MLOCK_MIN_FREE_AFTER_GB = _env_float("CIBOBUONO_MLOCK_MIN_FREE_AFTER_GB", 6.0)
ALLOW_MLOCK = _env_bool("CIBOBUONO_ALLOW_MLOCK", False)

# How nice (lower priority) the pipeline should be on POSIX.
NICE_INCREMENT = int(_env_float("CIBOBUONO_NICE", 10))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSnapshot:
    """Instantaneous view of free resources. All sizes in GiB."""

    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    swap_used_gb: float
    swap_used_percent: float
    load_per_core: float  # 1-min load average / logical cores (POSIX); else cpu%
    cpu_count: int
    gpu_total_gb: float | None
    gpu_free_gb: float | None
    gpu_used_percent: float | None

    def vram_pressure(self) -> bool:
        if self.gpu_total_gb and self.gpu_used_percent is not None:
            return self.gpu_used_percent >= PRESSURE_VRAM_PERCENT
        return False

    def summary(self) -> str:
        gpu = (
            f", VRAM {self.gpu_free_gb:.1f}/{self.gpu_total_gb:.1f} GB free"
            if self.gpu_total_gb
            else ""
        )
        return (
            f"RAM {self.ram_available_gb:.1f}/{self.ram_total_gb:.1f} GB free "
            f"({self.ram_used_percent:.0f}% used), swap {self.swap_used_percent:.0f}%, "
            f"load/core {self.load_per_core:.2f}{gpu}"
        )


def _gpustat_json() -> list[dict]:
    """Return gpustat JSON gpu list, or [] on any failure."""
    try:
        import json as _j
        out = subprocess.run(["gpustat", "--json"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return _j.loads(out.stdout).get("gpus") or []
    except Exception as exc:
        logger.debug("gpustat probe failed: %s", exc)
    return []


def _gpu_memory_gb() -> tuple[float | None, float | None, float | None]:
    """(total_gb, free_gb, used_percent) for the first CUDA GPU; nvidia-smi → gpustat fallback."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 2:
                total = float(parts[0]) / 1024.0
                free = float(parts[1]) / 1024.0
                pct = round((total - free) / total * 100.0, 1) if total > 0 else None
                return round(total, 2), round(free, 2), pct
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    except Exception as exc:
        logger.debug("nvidia-smi memory probe failed: %s", exc)

    gpus = _gpustat_json()
    if gpus:
        try:
            g = gpus[0]
            total_mb, used_mb = float(g["memory.total"]), float(g["memory.used"])
            total = total_mb / 1024.0
            free = max(0.0, (total_mb - used_mb) / 1024.0)
            pct = round(used_mb / total_mb * 100.0, 1) if total_mb > 0 else None
            return round(total, 2), round(free, 2), pct
        except (KeyError, TypeError, ValueError):
            pass
    return None, None, None


def _gpu_compute_percent() -> float | None:
    """GPU SM utilization %; nvidia-smi → gpustat fallback."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip().splitlines()[0].strip()), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    except Exception as exc:
        logger.debug("nvidia-smi utilization probe failed: %s", exc)

    gpus = _gpustat_json()
    if gpus:
        try:
            util = gpus[0].get("utilization.gpu")
            if util is not None:
                return round(float(util), 1)
        except (TypeError, ValueError):
            pass
    return None


def snapshot(*, include_gpu: bool = False) -> ResourceSnapshot:
    """Take an instantaneous resource reading.

    ``include_gpu`` triggers an nvidia-smi probe; skip it on non-CUDA hosts to
    avoid spawning a subprocess on every poll.
    """
    cpu_count = os.cpu_count() or 1

    ram_total = ram_avail = swap_used = 0.0
    ram_pct = swap_pct = 0.0
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        ram_total = vm.total / (1024**3)
        ram_avail = vm.available / (1024**3)
        ram_pct = float(vm.percent)
        swap_used = sm.used / (1024**3)
        swap_pct = float(sm.percent)
    except Exception:  # psutil missing or unavailable
        from scripts.hardware import get_profile

        ram_total = get_profile().total_ram_gb
        # Without psutil we cannot know what's free; assume a conservative half.
        ram_avail = ram_total * 0.5
        ram_pct = 50.0

    # CPU pressure: prefer load average (non-blocking) on POSIX.
    load_per_core = 0.0
    try:
        load1 = os.getloadavg()[0]
        load_per_core = load1 / max(1, cpu_count)
    except (OSError, AttributeError):
        try:
            import psutil  # type: ignore

            load_per_core = psutil.cpu_percent(interval=None) / 100.0
        except Exception:
            load_per_core = 0.0

    gpu_total = gpu_free = gpu_used_pct = None
    if include_gpu:
        gpu_total, gpu_free, gpu_used_pct = _gpu_memory_gb()

    return ResourceSnapshot(
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        ram_used_percent=round(ram_pct, 1),
        swap_used_gb=round(swap_used, 2),
        swap_used_percent=round(swap_pct, 1),
        load_per_core=round(load_per_core, 2),
        cpu_count=cpu_count,
        gpu_total_gb=gpu_total,
        gpu_free_gb=gpu_free,
        gpu_used_percent=gpu_used_pct,
    )


def dashboard_hardware() -> dict[str, float | None]:
    """Lightweight CPU/GPU % for the web dashboard (polled every second).

    GPU % is compute utilization (SM busy), matching nvtop — NOT VRAM usage.
    """
    cpu_pct: float | None = None
    try:
        import psutil  # type: ignore

        cpu_pct = float(psutil.cpu_percent(interval=0))
    except Exception:
        snap = snapshot(include_gpu=False)
        cpu_pct = min(100.0, snap.load_per_core / max(1, snap.cpu_count) * 100.0)

    gpu_pct = _gpu_compute_percent()
    _, _, vram_pct = _gpu_memory_gb()

    return {
        "cpu_percent": round(cpu_pct, 1) if cpu_pct is not None else None,
        "gpu_percent": gpu_pct,
        "gpu_memory_percent": vram_pct,
    }


# ---------------------------------------------------------------------------
# Pressure detection + back-pressure
# ---------------------------------------------------------------------------


def under_pressure(snap: ResourceSnapshot | None = None, *, include_gpu: bool = False) -> tuple[bool, str]:
    """Return (is_stressed, human_reason) for the current system state."""
    snap = snap or snapshot(include_gpu=include_gpu)
    reasons: list[str] = []
    if snap.ram_used_percent >= PRESSURE_RAM_PERCENT:
        reasons.append(f"RAM {snap.ram_used_percent:.0f}%≥{PRESSURE_RAM_PERCENT:.0f}%")
    if snap.swap_used_percent >= PRESSURE_SWAP_PERCENT:
        reasons.append(f"swap {snap.swap_used_percent:.0f}%≥{PRESSURE_SWAP_PERCENT:.0f}%")
    if snap.load_per_core >= PRESSURE_LOAD_PER_CORE:
        reasons.append(f"load/core {snap.load_per_core:.1f}≥{PRESSURE_LOAD_PER_CORE:.1f}")
    if snap.vram_pressure():
        reasons.append(f"VRAM {snap.gpu_used_percent:.0f}%≥{PRESSURE_VRAM_PERCENT:.0f}%")
    return bool(reasons), ", ".join(reasons)


def wait_until_calm(
    *,
    include_gpu: bool = False,
    timeout: float = PRESSURE_WAIT_SECONDS,
    should_abort=None,
) -> bool:
    """Block until the system is no longer under pressure or *timeout* elapses.

    ``should_abort`` is an optional zero-arg callable; when it returns True the
    wait stops early (used to honour graceful-shutdown requests).

    Returns True if the system is calm on return, False if we timed out (the
    caller should then degrade rather than pile on more load).
    """
    deadline = time.monotonic() + max(0.0, timeout)
    stressed, reason = under_pressure(include_gpu=include_gpu)
    if not stressed:
        return True
    logger.warning("System under pressure (%s); pausing before more work…", reason)
    while time.monotonic() < deadline:
        if should_abort is not None and should_abort():
            return False
        time.sleep(POLL_SECONDS)
        stressed, reason = under_pressure(include_gpu=include_gpu)
        if not stressed:
            logger.info("Pressure cleared; resuming.")
            return True
    logger.warning("Still under pressure after %.0fs (%s); proceeding cautiously.", timeout, reason)
    return False


# ---------------------------------------------------------------------------
# Adaptive model selection ("wait, then downgrade")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmPlan:
    """Concrete, hardware-safe plan for loading a GGUF model right now."""

    model_path: Path | None
    size_gb: float
    n_gpu_layers: int
    use_mlock: bool
    pool: str  # "VRAM" | "RAM"
    note: str


def list_gguf_models(models_dir: Path) -> list[tuple[Path, float]]:
    """Return [(path, size_gb)] for every *.gguf under *models_dir*, largest first."""
    out: list[tuple[Path, float]] = []
    try:
        for p in models_dir.glob("*.gguf"):
            try:
                out.append((p, p.stat().st_size / (1024**3)))
            except OSError:
                continue
    except OSError:
        return []
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def plan_llm_load(
    profile,
    models: list[tuple[Path, float]],
    *,
    should_abort=None,
) -> LlmPlan:
    """Pick the best GGUF that *currently* fits free memory.

    Strategy ("wait, then downgrade"):
      1. Determine the relevant memory pool — VRAM when the profile offloads to
         CUDA, otherwise system RAM.
      2. Filter to models the machine could *ever* hold (capacity ceiling).
      3. Prefer the largest such model. If it doesn't fit free memory right now,
         wait up to HEADROOM_WAIT_SECONDS for headroom.
      4. If it still doesn't fit, downgrade to the largest model that fits the
         currently-free memory; if none fits, fall back to the smallest one.
    """
    if not models:
        return LlmPlan(None, 0.0, 0, False, "RAM", "no GGUF models found")

    cuda_offload = profile.has_cuda and profile.n_gpu_layers != 0
    snap = snapshot(include_gpu=cuda_offload)

    if cuda_offload and snap.gpu_total_gb:
        pool, pool_total = "VRAM", snap.gpu_total_gb
        factor, min_free = VRAM_HEADROOM_FACTOR, MIN_FREE_VRAM_GB
        free_fn = lambda s: (s.gpu_free_gb or 0.0)
    else:
        pool, pool_total = "RAM", snap.ram_total_gb
        factor, min_free = RAM_HEADROOM_FACTOR, MIN_FREE_RAM_GB
        free_fn = lambda s: s.ram_available_gb

    def fits(size: float, free: float) -> bool:
        return size * factor + min_free <= free

    # Capacity ceiling: drop models that can't physically fit the whole pool.
    capable = [(p, s) for (p, s) in models if s * factor + min_free <= pool_total]
    if not capable:
        # Even the smallest exceeds capacity ceiling; take the smallest and warn.
        p, s = models[-1]
        return LlmPlan(
            p, s, _gpu_layers_for(profile, pool, s, free_fn(snap), fits),
            False, pool,
            f"all models exceed {pool} capacity ({pool_total:.1f} GB); using smallest {p.name}",
        )

    desired_path, desired_size = capable[0]  # largest that fits the pool at all

    # Does the largest desired model fit *free* memory now?
    if fits(desired_size, free_fn(snap)):
        return _build_plan(profile, pool, desired_path, desired_size, free_fn(snap), fits,
                           note=f"fits free {pool} ({free_fn(snap):.1f} GB)")

    # Wait for headroom, polling free memory.
    logger.warning(
        "%s desired model %s (%.1f GB) doesn't fit free %s (%.1f GB); waiting up to %.0fs…",
        pool, desired_path.name, desired_size, pool, free_fn(snap), HEADROOM_WAIT_SECONDS,
    )
    deadline = time.monotonic() + HEADROOM_WAIT_SECONDS
    while time.monotonic() < deadline:
        if should_abort is not None and should_abort():
            break
        time.sleep(POLL_SECONDS)
        snap = snapshot(include_gpu=cuda_offload)
        if fits(desired_size, free_fn(snap)):
            return _build_plan(profile, pool, desired_path, desired_size, free_fn(snap), fits,
                               note=f"headroom freed up for {pool}")

    # Downgrade: largest capable model that fits the currently-free memory.
    free_now = free_fn(snap)
    for p, s in capable:
        if fits(s, free_now):
            return _build_plan(profile, pool, p, s, free_now, fits,
                               note=f"downgraded to fit free {pool} ({free_now:.1f} GB)")

    # Nothing fits free memory: take the smallest capable model and hope mmap helps.
    p, s = capable[-1]
    return _build_plan(profile, pool, p, s, free_now, fits,
                       note=f"low memory ({free_now:.1f} GB free {pool}); smallest model, relying on mmap")


def _gpu_layers_for(profile, pool, size, free, fits) -> int:
    """Full GPU offload only when the model comfortably fits VRAM; else CPU."""
    if pool != "VRAM":
        return profile.n_gpu_layers  # CPU build or Metal (unified memory)
    return profile.n_gpu_layers if fits(size, free) else 0


def _build_plan(profile, pool, path, size, free, fits, *, note) -> LlmPlan:
    n_gpu_layers = _gpu_layers_for(profile, pool, size, free, fits)
    if n_gpu_layers == 0 and pool == "VRAM":
        note += " (CPU fallback: won't fit VRAM)"
    # mlock only when explicitly allowed AND generous headroom remains.
    free_after = free - size
    use_mlock = (
        ALLOW_MLOCK
        and profile.use_mlock
        and not profile.is_virtual
        and free_after >= MLOCK_MIN_FREE_AFTER_GB
    )
    return LlmPlan(path, round(size, 2), n_gpu_layers, use_mlock, pool, note)


# ---------------------------------------------------------------------------
# Whisper model fitting
# ---------------------------------------------------------------------------

# Approximate resident footprint (GiB) per Whisper size, largest first.
# Conservative (CT2 int8 / fp16 mix); used only to avoid loading a model that
# won't fit free memory. We never *upgrade* past the requested size.
WHISPER_LADDER: tuple[tuple[str, float], ...] = (
    ("large-v3-turbo", 3.0),
    ("large-v3", 3.2),
    ("large-v2", 3.2),
    ("large", 3.2),
    ("medium", 1.6),
    ("small", 0.6),
    ("base", 0.2),
    ("tiny", 0.1),
)


def fit_whisper_model(profile, requested: str) -> tuple[str, str]:
    """Return (model_name, note): the requested Whisper size, downgraded only if
    free memory can't hold it. On CUDA the pool is VRAM (shared with the LLM)."""
    names = [n for n, _ in WHISPER_LADDER]
    if requested not in names:
        return requested, "unknown size; left as requested"

    on_cuda = profile.whisper_device == "cuda"
    snap = snapshot(include_gpu=on_cuda)
    if on_cuda and snap.gpu_total_gb:
        free, factor, min_free, pool = (snap.gpu_free_gb or 0.0), VRAM_HEADROOM_FACTOR, MIN_FREE_VRAM_GB, "VRAM"
    else:
        free, factor, min_free, pool = snap.ram_available_gb, RAM_HEADROOM_FACTOR, MIN_FREE_RAM_GB, "RAM"

    start = names.index(requested)
    for name, size in WHISPER_LADDER[start:]:
        if size * factor + min_free <= free:
            if name == requested:
                return name, f"fits free {pool} ({free:.1f} GB)"
            return name, f"downgraded {requested}→{name} to fit free {pool} ({free:.1f} GB)"
    return WHISPER_LADDER[-1][0], f"very low {pool} ({free:.1f} GB); smallest Whisper model"


# ---------------------------------------------------------------------------
# Process priority
# ---------------------------------------------------------------------------


_priority_applied = False


def apply_friendly_priority() -> None:
    """Lower this process's scheduling priority so the OS stays responsive.

    Best-effort and idempotent: ``os.nice`` is cumulative, so we apply it at
    most once per process. Failures (no permission, unsupported platform) are
    logged at debug level and ignored.
    """
    global _priority_applied
    if _priority_applied:
        return
    _priority_applied = True
    try:
        if hasattr(os, "nice") and NICE_INCREMENT:
            os.nice(NICE_INCREMENT)
            logger.debug("Applied nice +%d", NICE_INCREMENT)
    except OSError as exc:  # pragma: no cover - platform dependent
        logger.debug("os.nice failed: %s", exc)
    try:  # Windows
        import psutil  # type: ignore

        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # pragma: no cover
        pass
