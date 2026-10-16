"""MMDetection 3.x adapter: ``init_detector`` and ``inference_detector`` as a Detector.

[Unverified against a live MMDetection until the environment spike: the plan's mmcv build
for the RTX 5060's sm_120 is still open, plan §5 Phase 0 item 4b.] The adapter uses only the
public ``mmdet.apis`` functions and the ``pred_instances`` fields (``bboxes``, ``scores``,
``labels``) of the returned data sample. Both modules are imported lazily through
:func:`import_module`, so tests inject fakes and the package imports without MMDetection.

``model.test_cfg`` overrides come from configuration (``score_threshold``, ``max_per_image``),
both required. The precision's numerics regime (:mod:`qcal_lab.numerics`) is applied before
the model is built, and the run records every switch in effect and the device it used.

The checkpoint's class names are read from the checkpoint itself before the model is built,
as MMDetection 3.3 reads them: ``init_detector`` substitutes COCO's names when a checkpoint
carries none, so the built model's names prove nothing. They must equal the dataset's, in
label order (exactly, or after folding case and underscores, as v1-era checkpoints spell
``traffic_light``). A checkpoint without names is refused unless
``detectors.<name>.label_order_attested_sha256`` names the dataset's category order: Ian's
attestation, after checking the label order by hand (D2).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
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
CHECKPOINT_MODULE: Final = "mmengine.runner"
#: The compiled ops whose build decides kernels; a source rebuild keeps mmcv's version string.
MMCV_EXTENSION: Final = "mmcv._ext"
ATTESTATION_SETTING: Final = "label_order_attested_sha256"
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


def label_order_digest(checkpoint_sha256: str, names: Sequence[str]) -> str:
    """The digest an attestation names: the checkpoint's sha256, then the dataset's category
    names, one per line, in order. It binds the attested order to that one checkpoint."""
    return hashlib.sha256("\n".join([checkpoint_sha256, *names]).encode("utf-8")).hexdigest()


def _folded(names: Sequence[str]) -> list[str]:
    return [str(n).casefold().replace("_", " ") for n in names]


def checkpoint_classes(loader: Any, checkpoint: Path) -> list[str] | None:
    """The class names a checkpoint carries, found where MMDetection 3.3 looks: its
    ``meta["dataset_meta"]`` (keys in any case), else ``meta["CLASSES"]``; ``None`` if absent.

    ``loader`` is ``mmengine.runner``: the deserialiser ``init_detector`` itself uses.
    """
    loaded = loader.CheckpointLoader.load_checkpoint(str(checkpoint), map_location="cpu")
    meta = loaded.get("meta") if isinstance(loaded, Mapping) else None
    if not isinstance(meta, Mapping):
        return None
    dataset_meta = meta.get("dataset_meta")
    if isinstance(dataset_meta, Mapping):
        lowered = {str(k).lower(): v for k, v in dataset_meta.items()}
        if lowered.get("classes") is not None:
            return [str(c) for c in lowered["classes"]]
    classes = meta.get("CLASSES")
    return None if classes is None else [str(c) for c in classes]


def checkpoint_label_map(
    name: str,
    classes: Sequence[str] | None,
    expected: Sequence[str],
    attested: Callable[[], str],
    checkpoint_sha256: str,
) -> str:
    """How the checkpoint's labels match the dataset's: ``exact``, ``folded`` or ``attested``.

    Refuses names that differ, and a checkpoint without names unless Ian attested its label
    order for exactly this checkpoint (:func:`label_order_digest`, in his declarations file;
    D2). ``attested`` is read only for such a checkpoint. The refusal never prints the value
    to set: it is Ian's to compute once he has checked the order.
    """
    if classes is None:
        if not checkpoint_sha256:
            raise DetectorError(
                f"{name}: the checkpoint carries no class names, and an attested label order "
                f"needs detectors.{name}.checkpoint_sha256 to name the checkpoint it is for"
            )
        if attested() == label_order_digest(checkpoint_sha256, expected):
            return "attested"
        raise DetectorError(
            f"{name}: the checkpoint carries no class names, so MMDetection would label it "
            f"with COCO's; Ian checks its label order and declares {ATTESTATION_SETTING} for "
            f"it (qcal_lab.models.mmdet.label_order_digest of checkpoint_sha256 and the "
            f"dataset's {len(expected)} category names; qcal_lab.declarations; D2)"
        )
    if list(classes) == list(expected):
        return "exact"
    if _folded(classes) == _folded(expected):
        return "folded"
    check_label_map(name, classes, expected)  # raises, naming the first differing label
    return "exact"  # pragma: no cover - check_label_map raised


def model_num_classes(model: Any) -> int | None:
    """How many classes a built model predicts: its (single- or two-stage) bbox head's
    ``num_classes``, else the length of its ``dataset_meta`` names; ``None`` if neither."""
    heads = (
        getattr(model, "bbox_head", None),
        getattr(getattr(model, "roi_head", None), "bbox_head", None),
    )
    for head in heads:
        count = getattr(head, "num_classes", None)
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    names = model_classes(model)
    return None if names is None else len(names)


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
        checkpoint_names: Sequence[str] | None = None,
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
        if checkpoint_names is not None:
            built = model_classes(self._model)
            if built != list(checkpoint_names):
                raise DetectorError(
                    f"{name}: the built model's classes differ from its checkpoint's "
                    f"({built!r} vs {list(checkpoint_names)!r})"
                )
        else:
            # An attested checkpoint has no names to compare: MMDetection labels the model
            # with COCO's, and Ian attested the label indices. What must still hold is the
            # count, or labels fall outside (or short of) the dataset's categories.
            count = model_num_classes(self._model)
            if count != ground_truth.num_classes:
                raise DetectorError(
                    f"{name}: the built model has {count if count is not None else 'an unknown'}"
                    f" number of classes, the dataset {ground_truth.num_classes}; an attested "
                    "label order needs the same number"
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
    loader = _import(CHECKPOINT_MODULE, "the checkpoint's class names cannot be read")
    try:
        names = checkpoint_classes(loader, checkpoint)
    except (  # a truncated or foreign file: torch.load raises any of these
        OSError,
        EOFError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        raise DetectorError(f"cannot read checkpoint {checkpoint}: {exc}") from exc
    label_map = checkpoint_label_map(
        context.name,
        names,
        context.ground_truth.category_names,
        lambda: str(context.declared().get(ATTESTATION_SETTING, "")),
        expected,
    )
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
            "label_map": label_map,
        },
        checkpoint_names=names,
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
        "mmcv_ext_sha256": mmcv_extension_digest(),
    }


def mmcv_extension_digest() -> str:
    """sha256 of mmcv's compiled ops (NMS and friends), ``""`` without them.

    A source rebuild for another GPU architecture keeps mmcv's version string, so the
    fingerprinted version alone would let the cache serve predictions from the old build.
    """
    try:
        spec = importlib.util.find_spec(MMCV_EXTENSION)
    except (ImportError, ValueError):
        return ""
    origin = None if spec is None else spec.origin
    return sha256_file(Path(origin)) if origin and Path(origin).is_file() else ""


DETECTORS.register("mmdet", _build)
KEY_PARTS.register("mmdet", key_parts)

__all__ = [
    "API_MODULE",
    "ATTESTATION_SETTING",
    "CHECKPOINT_MODULE",
    "CONFIG_MODULE",
    "MMCV_EXTENSION",
    "STACK_FIELDS",
    "TORCH_MODULE",
    "MMDetDetector",
    "check_label_map",
    "checkpoint_classes",
    "checkpoint_label_map",
    "key_parts",
    "label_order_digest",
    "mmcv_extension_digest",
    "model_classes",
    "model_num_classes",
]
