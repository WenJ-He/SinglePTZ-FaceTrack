# OPT3: 第三次实测优化

**日期**: 2026-04-15
**状态**: 已完成（待实测验证）

## 核心问题

OPT2让稳图和巡航时序正确了，但暴露出更致命的问题：`NET_DVR_PTZSelZoomIn_EX` 返回成功但球机根本没动，导致所有 SCAN_PICK 都是白工。

## 已完成任务

### OPT3-P0-C: 修复 identified 去重 bug

**根因**: `sim=0.128` 的单次低质量 STRANGER 识别被加入 identified 表，污染后续 3 个 preset 的所有 track 被跳过。

**修复**: `_add_identified()` 增加 `result.kind != "hit"` 守卫，只有确认命中库内人员才加入去重表。

### OPT3-P0-A1/A3: 诊断日志 + ISAPI 分辨率查询

- `zoom_to_bbox` 增加详细日志：归一化前后坐标、SDK 返回值、GetLastError 错误码
- `HikISAPI` 新增 `get_native_resolution()` 和 `get_streaming_resolution()` 方法
- 启动时自动查询并对比球机原生分辨率与 RTSP 流分辨率

### OPT3-P0-B: ISAPI 绝对定位替代方案（已验证可用）

- `HikISAPI` 新增 `ptz_absolute_zoom()` 方法，调用 `/ISAPI/PTZCtrl/channels/1/absoluteEx`
- `HikISAPI` 新增 `get_ptz_status()` 方法，读取当前 PTZ 绝对位置
- `HikPTZ._zoom_via_isapi()` 读取当前位置 → 计算 bbox 中心偏移 → 计算目标 zoom → 绝对定位
- `ptz_drag_zoom()` 已弃用（该球机不支持 ptzDrag 端点，返回 404）
- 切换方式：`config.yaml` 中 `ptz.zoom_backend: isapi`
- **实测验证**: ISAPI 绝对定位成功驱动球机（az 100.2→112.6, zoom 1.0→1.19），HTTP 200

### OPT3-P1: 下游逻辑修正

- **P1-B**: `_handle_scan_settle` 检测 motion 未触发时跳过 focus 判定，直接进 CAPTURE（不再浪费 100% 超时）
- **P1-C**: `_handle_scan_recognize` 样本不足 `min_samples_for_stranger`(3) 时判 UNRECOGNIZED 而非 STRANGER

### OPT3-P2: 诊断脚本

- 新增 `scripts/test_zoom_diagnosis.py`：支持 `--backend sdk/isapi/both` 分别测试 SDK 和 ISAPI 的 zoom 行为
- 自动查询分辨率、打印 SDK 返回值和错误码

### OPT3-P3: ISAPI zoom 稳图适配（关键修复）

**根因**: ISAPI zoom 是同步调用，球机在 `min_wait_after_zoom`(1.0s) 内完成移动并停下。
两阶段稳图检测（WAITING_MOTION→MOTION_DETECTED）在 min_wait 结束后才开始检查，
此时运动已结束，frame diff 始终低于 motion_th，永远无法进入 MOTION_DETECTED 阶段，
导致 `_handle_scan_settle` 误判为 "zoom 没生效"，跳过 focus 检查直接进 CAPTURE。

**修复**:
- `_record_ptz_cmd()`: 当 `zoom_backend == "isapi"` 且 `cmd_type == "zoom"` 时，
  跳过 `WAITING_MOTION` 阶段，直接设为 `MOTION_DETECTED`
- 稳图流程变为：min_wait(1.0s) → flush RTSP buffer → 直接检查帧稳定性 → focus 检查 → CAPTURE

### OPT3-P4: ISAPI zoom 填充比优化

**问题**: ISAPI zoom 目标填充比 0.6 (60%) 与 expand_ratio=1.5 组合过于保守。
人员检测 bbox 覆盖全身（高瘦），expand 后高度可超过帧高度，导致 zoom 反而缩小（<1.0x）。

**修复**: 将 `_zoom_via_isapi` 填充比从 0.6 提升至 0.85 (85%)，
确保 expand 后的 bbox 能获得有意义的放大倍数，同时保留足够边距给 capture tracking。

## 新增/修改配置

| 参数 | 值 | 说明 |
|------|------|------|
| `ptz.zoom_backend` | "isapi" | 3D定位后端，ISAPI 绝对定位已验证可用 |
| `recognize.min_samples_for_stranger` | 3 | 至少需要这么多样本才能判 STRANGER |
| ISAPI zoom fill ratio | 0.85 | 内置参数，bbox 填充帧比（原0.6） |

## 下一步操作

1. **重启实测**验证 ISAPI zoom 稳图流程：
   - 确认 settle 检测正确进入 MOTION_DETECTED 阶段
   - 确认 zoom 倍数合理（1.5x-3x+）
   - 确认 CAPTURE 阶段能成功采集 face crops
