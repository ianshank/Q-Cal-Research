"""MMDetection 3.x adapter: ``init_detector`` and ``inference_detector`` as a Detector.

[Unverified against a live MMDetection until the environment spike: the plan's mmcv build
for the RTX 5060's sm_120 is still open, plan §5 Phase 0 item 4b.] The adapter uses only the
public ``mmdet.apis`` functions and the ``pred_instances`` fields (``bboxes``, ``scores``,
``labels``) of the returned data sample. Both modules are imported lazily through
:func:`import_module`, so tests inject fakes and the package imports without MMDetection.

``model.test_cfg`` overrides come from configuration (``score_threshold``, ``max_per_image``);
omitted keys keep the MMDetection config's own values. ``precision = "fp32_tf32_off"``
disables TF32 for matmul and cuDNN before the model is built.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from qcal.log import get_logger
from qcal.protocols import Detection, ImageDetections
from qcal.registry.executor import sha256_file
from qcal_lab.data.coco import GroundTruth
from qcal_lab.models.base import DETECTORS, DetectorContext, DetectorError, setting_int

_log = get_logger("lab.models.mmdet")

API_MODULE: Final = "mmdet.apis"
TORCH_MODULE: Final = "torch"
TF32_OFF: Final = "fp32_tf32_off"
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


def set_tf32(torch: Any, *, enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def tf32_flags(torch: Any) -> dict[str, bool]:
    """The TF32 switches in effect, recorded with every run (plain fp32 keeps torch defaults)."""
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


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
    torch = _import(TORCH_MODULE, "the MMDetection detector cannot run")
    if context.precision == TF32_OFF:
        set_tf32(torch, enabled=False)
    api = _import(API_MODULE, "the MMDetection detector cannot run")
    return MMDetDetector(
        name=context.name,
        ground_truth=context.ground_truth,
        images_dir=context.images_dir,
        config_file=lab.file(f"{prefix}.config"),
        checkpoint=checkpoint,
        device=lab.text(f"{prefix}.device"),
        cfg_options=_cfg_options(context.settings),
        api=api,
        runtime={"precision": context.precision, **tf32_flags(torch)},
    )


DETECTORS.register("mmdet", _build)

__all__ = ["API_MODULE", "TF32_OFF", "MMDetDetector", "set_tf32", "tf32_flags"]
