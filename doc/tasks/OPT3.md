# OPT3: 第三次实测优化

**日期**: 2026-04-15
**状态**: 已完成（代码层面）

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

### OPT3-P0-B: ISAPI 3D定位替代方案

- `HikISAPI` 新增 `ptz_drag_zoom()` 方法，调用 `/ISAPI/PTZCtrl/channels/1/ptzDrag`
- `HikPTZ.zoom_to_bbox()` 增加 `zoom_backend` 分支：`sdk`（默认）或 `isapi`
- ISAPI 路径自动缩放 bbox 坐标到球机原生分辨率
- 切换方式：`config.yaml` 中 `ptz.zoom_backend: isapi`

### OPT3-P1: 下游逻辑修正

- **P1-B**: `_handle_scan_settle` 检测 motion 未触发时跳过 focus 判定，直接进 CAPTURE（不再浪费 100% 超时）
- **P1-C**: `_handle_scan_recognize` 样本不足 `min_samples_for_stranger`(3) 时判 UNRECOGNIZED 而非 STRANGER

### OPT3-P2: 诊断脚本

- 新增 `scripts/test_zoom_diagnosis.py`：支持 `--backend sdk/isapi/both` 分别测试 SDK 和 ISAPI 的 zoom 行为
- 自动查询分辨率、打印 SDK 返回值和错误码

## 新增/修改配置

| 参数 | 值 | 说明 |
|------|------|------|
| `ptz.zoom_backend` | "sdk" | 3D定位后端，可选 "sdk" 或 "isapi" |
| `recognize.min_samples_for_stranger` | 3 | 至少需要这么多样本才能判 STRANGER |

## 下一步操作

1. **运行诊断脚本**确认 zoom 行为：
   ```bash
   conda run -n single_ptz_facetrack python scripts/test_zoom_diagnosis.py
   conda run -n single_ptz_facetrack python scripts/test_zoom_diagnosis.py --backend isapi
   ```
2. 根据诊断结果决定 `ptz.zoom_backend` 配 `sdk` 还是 `isapi`
3. 如果 ISAPI 也失败，启用 P2-A（手动 PTZ 指令组合）
