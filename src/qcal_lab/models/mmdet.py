"""MMDetection 3.x adapter: ``init_detector`` and ``inference_detector`` as a Detector.

[Unverified against a live MMDetection until the environment spike: the plan's mmcv build
for the RTX 5060's sm_120 is still open, plan §5 Phase 0 item 4b.] The adapter uses only the
public ``mmdet.apis`` functions and the ``pred_instances`` fields (``bboxes``, ``scores``,
``labels``) of the returned data sample. Both modules are imported lazily through
:func:`import_module`, so tests inject fakes and the package imports without MMDetection.

``model.test_cfg`` overrides come from configuration (``score_threshold``, ``max_per_image``),
both required. The precision's numerics regime (:mod:`qcal_lab.numerics`) is applied before
the model is built, and the run records every switch in effect and the device it used. The
checkpoint's class names must equal the dataset's, in label order.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from qcal.log import get_logger
from qcal.protocols import Detection, ImageDetections
from qcal.registry.executor import sha256_file
from qcal_lab.config import LabConfig
from qcal_lab.data.coco import GroundTruth
from qcal_lab.models.base import (
    DETECTORS,
    KEY_PARTS,
    DetectorContext,
    DetectorError,
    setting_int,
)
from qcal_lab.numerics import apply_regime, device_record, effective_switches, load_regime

_log = get_logger("lab.models.mmdet")

API_MODULE: Final = "mmdet.apis"
TORCH_MODULE: Final = "torch"
CONFIG_MODULE: Final = "mmengine.config"
#: The device record fields that decide kernels, hence predictions (not which card it was).
STACK_FIELDS: Final = ("torch", "cuda", "cudnn", "torch_build_sha256", "capability")
_CFG_OPTIONS: Final = {
    "score_threshold": "model.test_cfg.score_thr",
    "max_per_image": "model.test_cfg.max_per_img",
}


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return list(value.tolist() if hasattr(value, "tolist") else value)


def _import(name: str, purpose: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise DetectorError(
            f"{name} is not importable ({exc}); {purpose}. See the environment spike in "
            "DECISIONS.md for the supported install."
        ) from exc


def check_label_map(name: str, classes: Sequence[str] | None, expected: Sequence[str]) -> None:
    """Refuse a model whose class names are absent or differ from the dataset's, in order.

    A detector's label ``i`` is read as the dataset's ``i``-th category, so a reordered or
    different label map would score every detection against the wrong class.
    """
    if classes is None:
        raise DetectorError(f"{name}: the model carries no class names; its labels are unchecked")
    if list(classes) == list(expected):
        return
    at = next(
        (i for i, (a, b) in enumerate(zip(classes, expected, strict=False)) if a != b),
        min(len(classes), len(expected)),
    )
    found = repr(classes[at]) if at < len(classes) else "nothing"
    wanted = repr(expected[at]) if at < len(expected) else "nothing"
    raise DetectorError(
        f"{name}: the model's classes differ from the dataset's at label {at} "
        f"({found}, expected {wanted}; {len(classes)} vs {len(expected)} classes)"
    )


def model_classes(model: Any) -> list[str] | None:
    """The class names a built MMDetection model carries (``dataset_meta``), if any."""
    meta = getattr(model, "dataset_meta", None)
    classes = meta.get("classes") if isinstance(meta, Mapping) else None
    return None if classes is None else [str(c) for c in classes]


class MMDetDetector:
    def __init__(
        self,
        *,
        name: str,
        ground_truth: GroundTruth,
        images_dir: Path,
        config_file: Path,
        checkpoint: Path,
        device: str,
        cfg_options: Mapping[str, Any],
        api: Any,
        runtime: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.runtime = dict(runtime or {})
        self.ground_truth = ground_truth
        self.images_dir = images_dir
        self._api = api
        self._model = api.init_detector(
            str(config_file),
            str(checkpoint),
            device=device,
            cfg_options=dict(cfg_options) or None,
        )
        check_label_map(name, model_classes(self._model), ground_truth.category_names)

    def _image(self, image_id: str) -> ImageDetections:
        info = self.ground_truth.images.get(image_id)
        if info is None:
            raise DetectorError(f"image {image_id} is not in the dataset")
        path = self.images_dir / info.file_name
        if not path.is_file():
            raise DetectorError(f"image file {path} does not exist")
        instances = self._api.inference_detector(self._model, str(path)).pred_instances
        boxes = _as_list(instances.bboxes)
        scores = _as_list(instances.scores)
        labels = _as_list(instances.labels)
        if not len(boxes) == len(scores) == len(labels):
            raise DetectorError(f"image {image_id}: bboxes, scores and labels differ in length")
        detections = [
            Detection((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(s), int(lab))
            for b, s, lab in zip(boxes, scores, labels, strict=True)
        ]
        detections.sort(key=lambda d: -d.score)
        return ImageDetections(image_id, tuple(detections))

    def predict(self, image_ids: Sequence[str]) -> list[ImageDetections]:
        return [self._image(i) for i in image_ids]


def _cfg_options(settings: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key, target in _CFG_OPTIONS.items():
        if key not in settings:
            continue
        if key == "max_per_image":
            options[target] = setting_int(settings, key)
        else:
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DetectorError(f"detector setting {key!r} must be a number")
            options[target] = float(value)
    return options


def _build(context: DetectorContext) -> MMDetDetector:
    lab, prefix = context.lab, f"detectors.{context.name}"
    if context.images_dir is None:
        raise DetectorError(f"{context.name}: MMDetection needs an images directory")
    checkpoint = lab.file(f"{prefix}.checkpoint")
    expected = str(context.settings.get("checkpoint_sha256", ""))
    if not checkpoint.is_file():
        raise DetectorError(f"checkpoint {checkpoint} does not exist")
    if expected:
        actual = sha256_file(checkpoint)
        if actual != expected:
            raise DetectorError(f"checkpoint {checkpoint} has sha256 {actual}, expected {expected}")
    else:
        _log.warning("%s: checkpoint_sha256 is not set; the checkpoint is not verified", prefix)
    regime = load_regime(lab, context.precision)
    torch = _import(TORCH_MODULE, "the MMDetection detector cannot run")
    numerics = apply_regime(torch, regime, os.environ)  # before the model is built
    api = _import(API_MODULE, "the MMDetection detector cannot run")
    device = lab.text(f"{prefix}.device")
    detector = MMDetDetector(
        name=context.name,
        ground_truth=context.ground_truth,
        images_dir=context.images_dir,
        config_file=lab.file(f"{prefix}.config"),
        checkpoint=checkpoint,
        device=device,
        cfg_options=_cfg_options(context.settings),
        api=api,
        runtime={
            "precision": context.precision,
            "numerics": numerics,
            "device": device_record(torch, device),
        },
    )
    # Importing mmdet and building the model may change switches; inference runs under these.
    detector.runtime["numerics"]["effective"] = effective_switches(torch)
    return detector


def _resolved_text(config: Any) -> str:
    """The resolved config as canonical JSON (sorted keys), so it does not depend on yapf's
    formatting the way ``pretty_text`` does; ``pretty_text`` only for a config without
    ``to_dict``."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return json.dumps(to_dict(), sort_keys=True, default=str)
    return str(config.pretty_text)


def key_parts(lab: LabConfig, name: str) -> dict[str, Any]:
    """The compute stack and the resolved model config, for the prediction cache key.

    The config file's own digest misses its ``_base_`` files, so the key holds the resolved
    config. Neither needs the model built, so a cache hit costs no detector.
    """
    prefix = f"detectors.{name}"
    torch = _import(TORCH_MODULE, "the MMDetection detector's cache key needs it")
    device = device_record(torch, lab.text(f"{prefix}.device"))
    config_file = lab.file(f"{prefix}.config")
    mmengine = _import(CONFIG_MODULE, "the MMDetection detector's cache key needs it")
    try:
        resolved = _resolved_text(mmengine.Config.fromfile(str(config_file)))
    except (
        OSError,
        SyntaxError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        AssertionError,
    ) as exc:
        raise DetectorError(f"cannot resolve the MMDetection config {config_file}: {exc}") from exc
    return {
        "stack": {field: device.get(field) for field in STACK_FIELDS},
        "resolved_config_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
    }


DETECTORS.register("mmdet", _build)
KEY_PARTS.register("mmdet", key_parts)

__all__ = [
    "API_MODULE",
    "CONFIG_MODULE",
    "STACK_FIELDS",
    "TORCH_MODULE",
    "MMDetDetector",
    "check_label_map",
    "key_parts",
    "model_classes",
]
