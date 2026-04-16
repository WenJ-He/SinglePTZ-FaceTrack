"""Stage-oriented config loader for the refactored runtime."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class RuntimeConfig:
    prefer_gpu: bool = False


@dataclass
class ModelConfig:
    person: str = "models/person/yolov8n.onnx"
    face: str = "models/facedect/640/model.onnx"


@dataclass
class DetectConfig:
    person_conf: float = 0.45
    person_iou: float = 0.5
    face_conf: float = 0.35
    face_iou: float = 0.5
    edge_reject_enabled: bool = True
    edge_margin: int = 5


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 18060
    jpeg_quality: int = 90


@dataclass
class LogConfig:
    level: str = "INFO"
    file: str = "logs/app.log"


@dataclass
class Stage1Config:
    home_preset: int = 1
    loop_sleep_s: float = 0.03
    settle_after_move_s: float = 0.35
    settle_after_zoom_s: float = 0.8
    hold_after_max_zoom_s: float = 2.0
    restore_home_settle_s: float = 2.5
    person_center_tolerance_ratio_x: float = 0.08
    person_center_tolerance_ratio_y: float = 0.10
    face_center_tolerance_ratio_x: float = 0.05
    face_center_tolerance_ratio_y: float = 0.07
    face_zoom_ready_ratio_x: float = 0.10
    face_zoom_ready_ratio_y: float = 0.12
    face_size_switch_ratio: float = 0.05
    face_size_hold_ratio: float = 0.12
    lost_timeout_s: float = 1.0
    move_speed: int = 4
    move_pulse_s: float = 0.18
    zoom_speed: int = 4
    zoom_pulse_s: float = 0.45
    min_zoom_interval_s: float = 0.30
    max_zoom_steps: int = 8
    debug_draw_all: bool = True


@dataclass
class Stage2Config:
    home_preset: int = 1
    loop_sleep_s: float = 0.03
    settle_after_move_s: float = 0.30
    settle_after_zoom_s: float = 0.80
    restore_home_settle_s: float = 2.5
    follow_duration_s: float = 10.0
    person_center_tolerance_ratio_x: float = 0.08
    person_center_tolerance_ratio_y: float = 0.10
    face_center_tolerance_ratio_x: float = 0.05
    face_center_tolerance_ratio_y: float = 0.07
    face_zoom_ready_ratio_x: float = 0.10
    face_zoom_ready_ratio_y: float = 0.12
    face_size_switch_ratio: float = 0.0
    body_anchor_ratio: float = 0.30
    face_predict_lead_s: float = 0.35
    body_predict_lead_s: float = 0.25
    lost_face_to_body_s: float = 0.5
    lost_all_restore_s: float = 1.0
    desired_face_ratio_min: float = 0.08
    desired_face_ratio_max: float = 0.16
    lost_timeout_s: float = 1.2
    move_speed_min: int = 2
    move_speed: int = 4
    move_pulse_min_s: float = 0.05
    move_pulse_s: float = 0.16
    move_control_interval_s: float = 0.08
    zoom_speed: int = 4
    zoom_pulse_s: float = 0.35
    min_zoom_interval_s: float = 0.50
    max_zoom_steps: int = 8
    debug_draw_all: bool = True


@dataclass
class AppConfig:
    stage: str = "stage1_single_static"
    hik: HikConfig = field(default_factory=HikConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    web: WebConfig = field(default_factory=WebConfig)
    log: LogConfig = field(default_factory=LogConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)


def _read_yaml(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_dataclass(cls, data):
    if not isinstance(data, dict):
        return data
    kwargs = {}
    for field_name, field_def in cls.__dataclass_fields__.items():
        if field_name not in data:
            continue
        value = data[field_name]
        field_type = field_def.type
        if isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__"):
            kwargs[field_name] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def load_app_config(
    common_path: str = "config/common.yaml",
    stage_path: str = "config/stage1_single_static.yaml",
) -> AppConfig:
    common = _read_yaml(common_path)
    stage = _read_yaml(stage_path)
    merged = _deep_update(common, stage)
    return _dict_to_dataclass(AppConfig, merged)


def auto_providers(prefer_gpu: bool = False):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
