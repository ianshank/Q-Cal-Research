"""Numerics regimes: the torch and cuDNN switches each precision runs under.

``[numerics.regimes.<precision>]`` names the switches a precision sets. A switch the regime
leaves out keeps torch's own default, and every switch in effect is recorded with the run,
set or not, with ``NVIDIA_TF32_OVERRIDE`` (which overrides TF32 beneath torch). ``fp32``
keeps torch's defaults; ``fp32_tf32_off`` turns TF32 and cuDNN autotuning off and
deterministic algorithms on. Which one G1 uses is Ian's decision D5; the default stays
``fp32``, and switching is one line of configuration.

torch is never imported here: callers that run torch pass it in, so the fixture detector and
the tests run without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from qcal.config import ConfigError
from qcal_lab.config import LabConfig

#: Boolean switches a regime may set, and where each lives under the torch module.
SWITCHES: Final[Mapping[str, tuple[str, ...]]] = {
    "allow_tf32_matmul": ("backends", "cuda", "matmul", "allow_tf32"),
    "allow_tf32_cudnn": ("backends", "cudnn", "allow_tf32"),
    "cudnn_benchmark": ("backends", "cudnn", "benchmark"),
    "cudnn_deterministic": ("backends", "cudnn", "deterministic"),
}
DETERMINISTIC_KEY: Final = "deterministic_algorithms"  # torch.use_deterministic_algorithms
WARN_ONLY_KEY: Final = "deterministic_warn_only"
REQUIRED_ENV_KEY: Final = "required_env"
_REGIME_KEYS: Final = frozenset({*SWITCHES, DETERMINISTIC_KEY, WARN_ONLY_KEY, REQUIRED_ENV_KEY})
#: Environment variables that change numerics beneath torch; recorded with every run.
RECORDED_ENV: Final = ("NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")


@dataclass(frozen=True)
class Regime:
    """The switches one precision sets; ``None`` and absent switches keep torch's defaults."""

    precision: str
    switches: Mapping[str, bool]
    deterministic: bool | None = None
    warn_only: bool = False
    required_env: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["switches"] = dict(self.switches)
        data["required_env"] = list(self.required_env)
        return data


def _flag(table: Mapping[str, Any], key: str, where: str) -> bool:
    value = table[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be true or false, got {value!r}")
    return value


def load_regime(lab: LabConfig, precision: str) -> Regime:
    """The regime of ``precision``; a precision without one is refused, as is an unknown key."""
    where = f"numerics.regimes.{precision}"
    regimes = lab.config.get("numerics", {}).get("regimes", {})
    table = regimes.get(precision) if isinstance(regimes, Mapping) else None
    if not isinstance(table, Mapping):
        raise ConfigError(f"precision {precision!r} has no [{where}] table")
    unknown = sorted(set(table) - _REGIME_KEYS)
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown} (known: {sorted(_REGIME_KEYS)})")
    required = table.get(REQUIRED_ENV_KEY, [])
    if not isinstance(required, list) or not all(isinstance(n, str) and n for n in required):
        raise ConfigError(f"{where}.{REQUIRED_ENV_KEY} must be a list of variable names")
    return Regime(
        precision=precision,
        switches={key: _flag(table, key, where) for key in SWITCHES if key in table},
        deterministic=_flag(table, DETERMINISTIC_KEY, where)
        if DETERMINISTIC_KEY in table
        else None,
        warn_only=_flag(table, WARN_ONLY_KEY, where) if WARN_ONLY_KEY in table else False,
        required_env=tuple(required),
    )


def _attribute(torch: Any, path: Sequence[str]) -> tuple[Any, str]:
    node = torch
    for part in path[:-1]:
        node = getattr(node, part)
    return node, path[-1]


def effective_switches(torch: Any) -> dict[str, bool]:
    """Every switch in effect, whether the regime set it or torch's default stands."""
    found: dict[str, bool] = {}
    for key, path in SWITCHES.items():
        node, name = _attribute(torch, path)
        found[key] = bool(getattr(node, name))
    found[DETERMINISTIC_KEY] = bool(torch.are_deterministic_algorithms_enabled())
    found[WARN_ONLY_KEY] = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    return found


def recorded_environment(environ: Mapping[str, str]) -> dict[str, str | None]:
    return {name: environ.get(name) for name in RECORDED_ENV}


def apply_regime(torch: Any, regime: Regime, environ: Mapping[str, str]) -> dict[str, Any]:
    """Set the regime's switches before a model is built; returns what to record."""
    missing = [name for name in regime.required_env if not environ.get(name)]
    if missing:
        raise ConfigError(
            f"precision {regime.precision!r} needs {', '.join(missing)} set in the environment "
            f"(numerics.regimes.{regime.precision}.{REQUIRED_ENV_KEY})"
        )
    for key, value in regime.switches.items():
        node, name = _attribute(torch, SWITCHES[key])
        setattr(node, name, value)
    if regime.deterministic is not None:
        torch.use_deterministic_algorithms(regime.deterministic, warn_only=regime.warn_only)
    return {
        "regime": regime.to_dict(),
        "effective": effective_switches(torch),
        "environment": recorded_environment(environ),
    }


def _cuda_index(device: str) -> int:
    _, _, index = device.partition(":")
    return int(index) if index.isdigit() else 0


def device_record(torch: Any, device: str) -> dict[str, Any]:
    """The device a run actually used and the CUDA stack beneath torch.

    ``nvidia-smi`` orders GPUs by PCI bus while CUDA orders them fastest first by default, so
    the launcher's inventory cannot say which GPU ran; the program records it here.
    """
    record: dict[str, Any] = {
        "device": device,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_build_sha256": hashlib.sha256(str(torch.__config__.show()).encode()).hexdigest(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(_cuda_index(device))
        record.update(
            name=str(props.name),
            uuid=str(getattr(props, "uuid", "")),
            capability=f"{props.major}.{props.minor}",
            total_memory=int(props.total_memory),
        )
    return record


__all__ = [
    "DETERMINISTIC_KEY",
    "RECORDED_ENV",
    "REQUIRED_ENV_KEY",
    "SWITCHES",
    "WARN_ONLY_KEY",
    "Regime",
    "apply_regime",
    "device_record",
    "effective_switches",
    "load_regime",
    "recorded_environment",
]
