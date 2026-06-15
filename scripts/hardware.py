"""
hardware.py — Cross-platform hardware profile detection.

Detects the runtime environment (Apple Silicon, Linux CUDA / ROCm, Raspberry Pi,
Mac Intel, Windows, generic ARM/x86 Linux, container / VM) and produces a
frozen :class:`DeviceProfile` with optimal parameters for:

- faster-whisper:  ``device`` ("cuda" | "cpu"), ``compute_type``
  ("float16" | "int8_float16" | "int8"), ``cpu_threads``
- llama.cpp / llama-cpp-python:  ``n_threads``, ``n_gpu_layers``, ``n_batch``,
  ``n_ctx``, ``use_mlock``, ``use_mmap``
- model tiering:  Whisper model size + GGUF filename fragment

Best-practice references baked into the heuristics:
- faster-whisper has NO Metal backend; on Apple Silicon it runs CPU-only with
  int8 quantization (SYSTRAN/faster-whisper README, 2026).
- CUDA: float16 is the default; int8_float16 if VRAM < 8 GB.
- llama-cpp-python on Apple Silicon: n_gpu_layers=-1 with Metal build, Flash
  Attention + Q8_0 KV cache for ~20-30% memory savings.
- Raspberry Pi: stick to Q4_K_M GGUF, small context, no mlock.
- VMs / containers: mlock often fails due to RLIMIT_MEMLOCK; we disable it.

Detection avoids spawning subprocesses where possible (uses ctypes / sysfs /
procfs) so the profile cost is negligible at startup.

Scope: this module reports the machine's *total capacity* and the best
parameters that capacity could ever support. It is cached once per process and
never changes. The *live* allocation — choosing what to load given how much RAM
/ VRAM is free right now, throttling under pressure, dropping process priority —
is handled by :mod:`scripts.resource_monitor`, which treats the values here as
ceilings.
"""

from __future__ import annotations

__author__ = "Luca Ostinelli"

import json
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("hardware")


# ---------------------------------------------------------------------------
# Enums + dataclass
# ---------------------------------------------------------------------------


class DevicePlatform(str, Enum):
    APPLE_SILICON = "apple_silicon"
    MAC_INTEL = "mac_intel"
    LINUX_CUDA = "linux_cuda"
    LINUX_ROCM = "linux_rocm"
    LINUX_ARM64 = "linux_arm64"
    LINUX_X86_64 = "linux_x86_64"
    RASPBERRY_PI = "raspberry_pi"
    WINDOWS_CUDA = "windows_cuda"
    WINDOWS_CPU = "windows_cpu"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable snapshot of the detected hardware and the derived runtime
    parameters for faster-whisper and llama-cpp-python.
    """

    # Platform / OS
    platform: DevicePlatform
    system: str  # "Darwin" | "Linux" | "Windows"
    machine: str  # "arm64" | "x86_64" | "aarch64" | "armv7l"

    # CPU
    cpu_count_logical: int
    cpu_count_physical: int
    cpu_perf_cores: int  # Apple Silicon P-cores; else == physical

    # Memory
    total_ram_gb: float

    # Virtualization / container
    is_virtual: bool
    virt_type: str | None  # "kvm" | "docker" | "wsl" | "lxc" | "qemu" | …

    # Raspberry Pi
    pi_model: str | None  # full string from /proc/device-tree/model

    # GPU
    has_metal: bool
    has_cuda: bool
    has_rocm: bool
    gpu_name: str | None
    gpu_vram_gb: float | None

    # Derived: faster-whisper
    whisper_device: str  # "cuda" | "cpu"
    whisper_compute_type: str  # "float16" | "int8_float16" | "int8"
    whisper_cpu_threads: int
    whisper_model: str  # "large-v3-turbo" | "medium" | "small" | "tiny"

    # Derived: llama.cpp
    n_threads: int
    n_gpu_layers: int  # -1 for full offload, 0 for CPU-only
    n_batch: int
    n_ctx: int
    use_mlock: bool
    use_mmap: bool

    # Model tiering
    llm_tier: str  # "72B" | "32B" | "14B" | "8B" | "3B" | "1B" | "none"
    enable_llm: bool  # False on very low-end → NER+rules-only extraction

    # Provenance
    detection_notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = self.platform.value
        d["detection_notes"] = list(self.detection_notes)
        return d


# ---------------------------------------------------------------------------
# CPU / RAM detection
# ---------------------------------------------------------------------------


def _detect_physical_cpu_count(logical: int) -> int:
    """Physical core count, cross-platform.

    Prefers psutil when available; falls back to os.cpu_count() (logical)."""
    try:
        import psutil  # type: ignore

        n = psutil.cpu_count(logical=False)
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    return logical


def _detect_apple_silicon_cores() -> tuple[int, int]:
    """Return (performance_cores, efficiency_cores) on Apple Silicon.

    Uses ctypes + sysctlbyname so we don't fork a subprocess. The hw.perflevel
    sysctls are stable since macOS 11 (M1).
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.sysctlbyname.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        libc.sysctlbyname.restype = ctypes.c_int

        def _read(key: str) -> int:
            val = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(val))
            rc = libc.sysctlbyname(
                key.encode("utf-8"),
                ctypes.byref(val),
                ctypes.byref(size),
                None,
                0,
            )
            if rc != 0:
                raise OSError(f"sysctlbyname({key}) failed: rc={rc}")
            return int(val.value)

        p = _read("hw.perflevel0.physicalcpu")
        try:
            e = _read("hw.perflevel1.physicalcpu")
        except OSError:
            # No efficiency cores reported (rare; treat as 0)
            e = 0
        return p, e
    except Exception as exc:
        logger.debug("_detect_apple_silicon_cores failed: %s", exc)
        # Conservative fallback: half logical as performance, half as efficiency
        n = os.cpu_count() or 4
        return max(1, n // 2), max(0, n - n // 2)


def _detect_ram_gb() -> float:
    """Total system RAM in GiB, cross-platform."""
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass

    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return round(int(out.stdout.strip()) / (1024**3), 2)
        elif sys_name == "Linux":
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            return round(page_size * phys_pages / (1024**3), 2)
        elif sys_name == "Windows":
            out = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in out.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    return round(int(line) / (1024**3), 2)
    except Exception as exc:
        logger.debug("_detect_ram_gb fallback failed: %s", exc)
    return 8.0  # safe default


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def _detect_cuda() -> tuple[bool, str | None, float | None]:
    """(has_cuda, gpu_name, vram_gb).

    Tries nvidia-smi first; falls back to PyTorch CUDA detection when
    nvidia-smi is not installed (driver present but tool missing).
    Returns (False, None, None) when no NVIDIA GPU is found.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = None
    except Exception as exc:
        logger.debug("nvidia-smi probe failed: %s", exc)
        out = None

    if out is not None and out.returncode == 0 and out.stdout.strip():
        first = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            name = parts[0] or None
            try:
                vram_mb = float(parts[1])
                vram_gb = round(vram_mb / 1024.0, 2)
            except ValueError:
                vram_gb = None
            return True, name, vram_gb

    # Fallback: PyTorch CUDA detection (works when nvidia-smi is missing)
    try:
        import torch  # lazy import — not in requirements-ci.txt
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / 1024 ** 3, 2)
            logger.debug("CUDA detected via PyTorch: %s (%.2f GB)", name, vram_gb)
            return True, name, vram_gb
    except Exception as exc:
        logger.debug("PyTorch CUDA fallback failed: %s", exc)

    return False, None, None


def _detect_rocm() -> tuple[bool, str | None, float | None]:
    """Best-effort ROCm detection on Linux."""
    if platform.system() != "Linux":
        return False, None, None
    if Path("/sys/module/amdgpu").exists() or Path("/dev/kfd").exists():
        # Try rocm-smi for VRAM, but don't require it.
        name: str | None = "AMD GPU"
        vram_gb: float | None = None
        try:
            out = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0:
                for line in out.stdout.splitlines():
                    if "Card series" in line or "Card model" in line:
                        name = line.split(":", 1)[-1].strip() or name
                    m = re.search(r"vram.*Total.*:\s*(\d+)", line, re.IGNORECASE)
                    if m:
                        vram_gb = round(int(m.group(1)) / (1024**3), 2)
        except Exception:
            # rocm-smi missing/older format/timeout — VRAM stays unknown.
            pass
        return True, name, vram_gb
    return False, None, None


# ---------------------------------------------------------------------------
# Raspberry Pi detection
# ---------------------------------------------------------------------------


_PI_MODEL_PATH = Path("/proc/device-tree/model")


def _detect_raspberry_pi() -> str | None:
    """Return the full Pi model string ('Raspberry Pi 5 Model B Rev 1.0') or None."""
    if not _PI_MODEL_PATH.exists():
        return None
    try:
        raw = _PI_MODEL_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    model = raw.replace("\x00", "").strip()
    if model.lower().startswith("raspberry pi"):
        return model
    return None


# ---------------------------------------------------------------------------
# Virtualization / container detection
# ---------------------------------------------------------------------------


_CGROUP_HINTS = ("docker", "kubepods", "lxc", "containerd", "podman", "crio")


def _detect_virtualization() -> tuple[bool, str | None]:
    """Cascade detection. Returns (is_virtual, virt_type)."""
    sys_name = platform.system()

    # 1. systemd-detect-virt (most authoritative on Linux)
    if sys_name == "Linux":
        try:
            out = subprocess.run(
                ["systemd-detect-virt"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            label = out.stdout.strip()
            if out.returncode == 0 and label and label != "none":
                return True, label
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as exc:
            logger.debug("systemd-detect-virt failed: %s", exc)

        # 2. Docker / Podman / Kubernetes sentinel files
        if Path("/.dockerenv").exists():
            return True, "docker"
        if Path("/run/.containerenv").exists():
            return True, "podman"
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return True, "kubernetes"

        # 3. /proc/self/cgroup hints
        try:
            cg = Path("/proc/self/cgroup").read_text(encoding="utf-8")
            for hint in _CGROUP_HINTS:
                if hint in cg:
                    return True, hint
        except OSError:
            pass

        # 4. WSL
        try:
            ver = Path("/proc/version").read_text(encoding="utf-8").lower()
            if "microsoft" in ver or "wsl" in ver:
                return True, "wsl"
        except OSError:
            pass

        # 5. hypervisor flag in /proc/cpuinfo
        try:
            ci = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in ci.splitlines():
                if line.startswith("flags") and " hypervisor" in line:
                    return True, "hypervisor"
        except OSError:
            pass

    # 6. macOS — sysctl machdep.cpu.features contains "VMM" when running under
    #    Apple's Virtualization framework or Parallels/VMware.
    if sys_name == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0 and "VMM" in out.stdout.upper():
                return True, "macos-hypervisor"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 7. Windows — best-effort env vars
    if sys_name == "Windows":
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return True, "kubernetes"
        if os.environ.get("CONTAINER_SANDBOX_MOUNT_POINT"):
            return True, "windows-container"

    return False, None


# ---------------------------------------------------------------------------
# Platform classification
# ---------------------------------------------------------------------------


def _classify_platform(
    system: str,
    machine: str,
    has_cuda: bool,
    has_rocm: bool,
    pi_model: str | None,
) -> DevicePlatform:
    if system == "Darwin":
        return DevicePlatform.APPLE_SILICON if machine == "arm64" else DevicePlatform.MAC_INTEL
    if system == "Windows":
        return DevicePlatform.WINDOWS_CUDA if has_cuda else DevicePlatform.WINDOWS_CPU
    if system == "Linux":
        if pi_model is not None:
            return DevicePlatform.RASPBERRY_PI
        if has_cuda:
            return DevicePlatform.LINUX_CUDA
        if has_rocm:
            return DevicePlatform.LINUX_ROCM
        if machine in {"aarch64", "armv7l", "armv6l", "arm64"}:
            return DevicePlatform.LINUX_ARM64
        return DevicePlatform.LINUX_X86_64
    return DevicePlatform.UNKNOWN


# ---------------------------------------------------------------------------
# Parameter derivation per platform
# ---------------------------------------------------------------------------


def _derive_params(
    plat: DevicePlatform,
    *,
    cpu_count_logical: int,
    cpu_count_physical: int,
    cpu_perf_cores: int,
    total_ram_gb: float,
    is_virtual: bool,
    gpu_vram_gb: float | None,
    pi_model: str | None,
) -> dict:
    """Return a dict of derived runtime params. Values are clamped to safe
    minimums so the pipeline never tries to spawn 0 threads etc."""

    # ── faster-whisper defaults ───────────────────────────────────────────
    whisper_device = "cpu"
    whisper_compute_type = "int8"
    whisper_cpu_threads = max(1, cpu_count_physical - 1)
    whisper_model = "large-v3-turbo"

    # ── llama.cpp defaults ────────────────────────────────────────────────
    n_threads = max(2, cpu_count_physical - 1)
    n_gpu_layers = 0
    n_batch = 512
    n_ctx = 4096
    use_mlock = False
    use_mmap = True

    # ── LLM tier defaults (RAM-bucketed) ──────────────────────────────────
    enable_llm = total_ram_gb >= 1.5

    if plat == DevicePlatform.APPLE_SILICON:
        # faster-whisper has no Metal backend → CPU + int8 + P-core threads
        whisper_cpu_threads = max(2, cpu_perf_cores)
        # llama.cpp w/ Metal handles GPU offload via build flag, n_gpu_layers=-1
        n_threads = max(2, cpu_perf_cores)
        n_gpu_layers = -1
        n_batch = 2048 if total_ram_gb >= 16 else 1024
        n_ctx = 8192 if total_ram_gb >= 16 else 4096
        use_mlock = not is_virtual

    elif plat in (DevicePlatform.LINUX_CUDA, DevicePlatform.WINDOWS_CUDA):
        whisper_device = "cuda"
        if gpu_vram_gb is not None and gpu_vram_gb >= 8:
            whisper_compute_type = "float16"
        else:
            whisper_compute_type = "int8_float16"
        # CTranslate2 doesn't use cpu_threads on CUDA; set sensibly anyway.
        whisper_cpu_threads = max(1, cpu_count_physical - 1)
        n_threads = max(2, cpu_count_physical - 1)
        n_gpu_layers = -1
        n_batch = 2048
        # With flash attention + Q8_0 KV cache, 16384 context on a 14B model
        # costs ~1.3 GB VRAM. Require ≥12 GB so 8 GB cards keep the safe 8192.
        n_ctx = 16384 if (gpu_vram_gb is not None and gpu_vram_gb >= 12) else 8192
        use_mlock = not is_virtual
        if gpu_vram_gb is not None and gpu_vram_gb < 4:
            whisper_model = "medium"

    elif plat == DevicePlatform.LINUX_ROCM:
        # faster-whisper has no native ROCm path → run CPU, but let llama.cpp
        # try to use ROCm via its build flags (n_gpu_layers=-1).
        whisper_device = "cpu"
        whisper_compute_type = "int8"
        n_threads = max(2, cpu_count_physical - 1)
        n_gpu_layers = -1
        n_batch = 2048
        n_ctx = 8192
        use_mlock = not is_virtual

    elif plat == DevicePlatform.RASPBERRY_PI:
        # Pi-tier params — see README hardware support table.
        whisper_cpu_threads = max(2, min(4, cpu_count_physical))
        n_threads = whisper_cpu_threads
        n_gpu_layers = 0
        n_batch = 256
        use_mmap = True
        use_mlock = False
        if total_ram_gb >= 6:  # Pi 5 / 8 GB
            whisper_model = "small"
            n_ctx = 2048
        elif total_ram_gb >= 3:  # Pi 4 / 4 GB
            whisper_model = "small"
            n_ctx = 1024
        else:  # Pi 3 / Zero 2W / 1 GB or less
            whisper_model = "tiny"
            n_ctx = 1024
            enable_llm = False

    elif plat == DevicePlatform.LINUX_ARM64:
        # Generic ARM Linux SBC / cloud (e.g. Ampere, AWS Graviton)
        whisper_compute_type = "int8"
        n_threads = max(2, cpu_count_physical - 1)
        n_batch = 512
        n_ctx = 4096
        if total_ram_gb < 4:
            whisper_model = "small"

    elif plat in (DevicePlatform.LINUX_X86_64, DevicePlatform.WINDOWS_CPU, DevicePlatform.MAC_INTEL):
        whisper_compute_type = "int8"
        n_threads = max(2, cpu_count_physical - 1)
        n_batch = 512
        n_ctx = 4096
        if total_ram_gb < 8:
            whisper_model = "medium"

    # ── VM / container global adjustments ─────────────────────────────────
    if is_virtual:
        use_mlock = False  # mlock typically blocked by RLIMIT_MEMLOCK
        n_threads = max(2, n_threads - 1)  # leave a core for the host

    # ── LLM tier picking ──────────────────────────────────────────────────
    llm_tier = _pick_llm_tier(total_ram_gb, enable_llm)
    if llm_tier == "none":
        enable_llm = False

    return dict(
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        whisper_cpu_threads=whisper_cpu_threads,
        whisper_model=whisper_model,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=n_ctx,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        llm_tier=llm_tier,
        enable_llm=enable_llm,
    )


# ---------------------------------------------------------------------------
# LLM tiering
# ---------------------------------------------------------------------------


# (filename_fragment, min_ram_gb).  Larger model first so we pick the best
# that fits.  Single source of truth — re-exported by scripts.utils as
# LLM_TIER_CANDIDATES for legacy callers.
LLM_TIERS: tuple[tuple[str, float], ...] = (
    ("72B", 40),
    ("70B", 40),
    ("32B", 24),
    ("27B", 20),
    ("14B", 12),
    ("8B", 6),
    ("3B", 3),
    ("1B", 1.5),
)


def _pick_llm_tier(ram_gb: float, enable_llm: bool) -> str:
    if not enable_llm:
        return "none"
    for tier, min_ram in LLM_TIERS:
        if ram_gb >= min_ram:
            return tier
    return "none"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_profile() -> DeviceProfile:
    """Detect the runtime environment exactly once per process.

    Cached via lru_cache so all callers share the same frozen profile.
    """
    notes: list[str] = []
    system = platform.system()
    machine = platform.machine()

    # CPU
    cpu_logical = os.cpu_count() or 4
    cpu_physical = _detect_physical_cpu_count(cpu_logical)
    if system == "Darwin" and machine == "arm64":
        p_cores, e_cores = _detect_apple_silicon_cores()
        perf_cores = max(1, p_cores)
        # On Apple Silicon, physical CPU = P + E
        if p_cores + e_cores > 0:
            cpu_physical = p_cores + e_cores
        notes.append(f"apple_silicon_cores: {p_cores}P+{e_cores}E")
    else:
        perf_cores = cpu_physical

    # Memory
    total_ram_gb = _detect_ram_gb()

    # Virtualization
    is_virtual, virt_type = _detect_virtualization()
    if is_virtual:
        notes.append(f"virt={virt_type}")

    # GPUs
    has_metal = system == "Darwin" and machine == "arm64"
    has_cuda, gpu_name, gpu_vram_gb = _detect_cuda()
    has_rocm = False
    if not has_cuda:
        has_rocm, rocm_name, rocm_vram = _detect_rocm()
        if has_rocm:
            gpu_name, gpu_vram_gb = rocm_name, rocm_vram
    if has_cuda:
        notes.append(f"cuda: {gpu_name} ({gpu_vram_gb} GB)")
    elif has_rocm:
        notes.append(f"rocm: {gpu_name}")
    elif has_metal:
        notes.append("metal")

    # Raspberry Pi
    pi_model = _detect_raspberry_pi() if system == "Linux" else None
    if pi_model:
        notes.append(f"pi_model={pi_model}")

    # Platform classification
    plat = _classify_platform(system, machine, has_cuda, has_rocm, pi_model)

    # Derived params
    derived = _derive_params(
        plat,
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        cpu_perf_cores=perf_cores,
        total_ram_gb=total_ram_gb,
        is_virtual=is_virtual,
        gpu_vram_gb=gpu_vram_gb,
        pi_model=pi_model,
    )

    profile = DeviceProfile(
        platform=plat,
        system=system,
        machine=machine,
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        cpu_perf_cores=perf_cores,
        total_ram_gb=total_ram_gb,
        is_virtual=is_virtual,
        virt_type=virt_type,
        pi_model=pi_model,
        has_metal=has_metal,
        has_cuda=has_cuda,
        has_rocm=has_rocm,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        detection_notes=tuple(notes),
        **derived,
    )

    logger.info(
        "Hardware profile: %s (%s, %dP+%dE cores, %.1f GB RAM%s); "
        "Whisper=%s (%s,%s); LLM=%s (n_gpu_layers=%d, n_threads=%d)",
        profile.platform.value,
        profile.machine,
        profile.cpu_perf_cores,
        max(0, profile.cpu_count_physical - profile.cpu_perf_cores),
        profile.total_ram_gb,
        f", virt={profile.virt_type}" if profile.is_virtual else "",
        profile.whisper_model,
        profile.whisper_device,
        profile.whisper_compute_type,
        profile.llm_tier,
        profile.n_gpu_layers,
        profile.n_threads,
    )

    return profile


def reset_profile_cache() -> None:
    """Drop the cached profile. Used by tests; not part of the public API."""
    get_profile.cache_clear()


# ---------------------------------------------------------------------------
# CLI: `python -m scripts.hardware` prints the JSON profile.
# ---------------------------------------------------------------------------


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    profile = get_profile()
    json.dump(profile.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
