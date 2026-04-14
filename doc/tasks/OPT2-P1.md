# OPT2-P1: 消除误检源

**批次**: OPT2-P1 (P1 必须改)
**日期**: 2026-04-14
**状态**: 已完成

## 问题描述

日志中30个检测样本有27个x1=0（90%贴左边缘），坐标反复稳定在同一位置(0, 600-660, 170-185, 800-830)，说明画面左侧固定物体被持续误检为人脸。

## 包含任务

### OPT2-P1-1: 过滤贴边误检bbox

**修改文件**: `src/detect/yolo_face.py`, `src/utils/geometry.py`, `src/config.py`, `config/config.yaml`, `src/main.py`

- `geometry.py` 新增 `is_edge_bbox()` 辅助函数：判断bbox是否贴边（距画面边界<margin像素）
- `yolo_face.py` 的 `detect()` 方法在NMS之后增加边缘过滤：贴边bbox直接丢弃
- 边缘过滤仅对 `face_wide`(1280) 启用，`face_close`(640) 不启用
- 新增配置项 `detect.edge_reject_enabled` 和 `detect.edge_margin`

### OPT2-P1-3: 真正接入人体检测做巡航触发

**状态**: 已在上一轮优化中实现

- `_handle_patrol_dwell` 和 `_handle_scan_detect` 已优先使用 `person_det`，仅未加载时 fallback 到 `face_wide`
- `person_conf` 已设为 0.45

## 新增配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `detect.edge_reject_enabled` | true | 是否启用贴边bbox过滤 |
| `detect.edge_margin` | 5 | bbox距画面边界<5px视为贴边 |

## 预期效果

- 日志中 x1=0 的误检消失
- SCAN模式不再因画面左侧固定物体被误触发
