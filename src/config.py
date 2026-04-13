"""YAML config -> dataclass mapping."""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class HikConfig:
    ip: str = "192.168.0.249"
    port: int = 8000
    user: str = "admin"
    password: str = ""
    channel: int = 1
    rtsp_url: str = ""
    sdk_lib_dir: str = ""


@dataclass
class PatrolConfig:
    presets: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    dwell: float = 4.0
    min_confirm_frames: int = 2


@dataclass
class ModelConfig:
    face_wide: str = "models/facedect/1280/best.onnx"
    face_close: str = "models/facedect/640/model.onnx"
    arcface: str = "models/facerecognize/model.onnx"
    reid: str = "models/reid/osnet_x0_25.onnx"
    person: str = "models/person/yolov8n.onnx"


@dataclass
class RuntimeConfig:
    prefer_gpu: bool = True


@dataclass
class DetectConfig:
    face_wide_conf: float = 0.50
    face_wide_iou: float = 0.5
    face_close_conf: float = 0.35
    face_close_iou: float = 0.5
    person_conf: float = 0.45
    person_iou: float = 0.5


@dataclass
class TrackConfig:
    iou_weight: float = 0.6
    reid_weight: float = 0.4
    max_age: int = 30
    min_hits: int = 3
    reid_ema: float = 0.1


@dataclass
class PtzConfig:
    expand_ratio: float = 1.5
    settle_diff_th: float = 8.0
    settle_timeout: float = 2.0


@dataclass
class CaptureTrackingConfig:
    enabled: bool = True
    safe_zone_ratio: float = 0.6
    correction_settle: float = 0.5
    max_corrections: int = 3
    face_lost_kalman_ms: int = 500
    face_lost_giveup_ms: int = 1200


@dataclass
class CaptureConfig:
    min_samples: int = 3
    max_samples: int = 5
    timeout: float = 4.0
    tracking: CaptureTrackingConfig = field(default_factory=CaptureTrackingConfig)


@dataclass
class RecognizeConfig:
    match_th: float = 0.35
    reject_th: float = 0.20


@dataclass
class ReidConfig:
    cross_preset_th: float = 0.5


@dataclass
class DisplayConfig:
    mode: str = "web"
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    jpeg_quality: int = 90


@dataclass
class OutputConfig:
    strangers_dir: str = "output/strangers"
    events_jsonl: str = "output/events.jsonl"


@dataclass
class LogConfig:
    level: str = "INFO"
    file: str = "logs/app.log"


@dataclass
class AppConfig:
    hik: HikConfig = field(default_factory=HikConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    ptz: PtzConfig = field(default_factory=PtzConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    recognize: RecognizeConfig = field(default_factory=RecognizeConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log: LogConfig = field(default_factory=LogConfig)


def _dict_to_dataclass(cls, d):
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    if not isinstance(d, dict):
        return d
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        ftype = field_types[k]
        # Handle nested dataclass
        origin = getattr(ftype, "__origin__", None)
        if isinstance(ftype, type) and hasattr(ftype, "__dataclass_fields__"):
            kwargs[k] = _dict_to_dataclass(ftype, v)
        elif origin is list:
            kwargs[k] = v
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """Load YAML config file and return AppConfig dataclass."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(AppConfig, raw) if raw else AppConfig()


def auto_providers(prefer_gpu: bool = True):
    """Detect available ONNX Runtime execution providers.

    Returns list of providers to try, preferring CUDA if available.
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if prefer_gpu:
        import logging
        logging.getLogger("app").info(
            "CUDAExecutionProvider not available, falling back to CPU"
        )
    return ["CPUExecutionProvider"]
