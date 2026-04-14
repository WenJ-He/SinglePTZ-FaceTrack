# OPT2-P0: PTZ异步等待核心修正

**批次**: OPT2-P0 (P0 必须改)
**日期**: 2026-04-14
**状态**: 已完成

## 问题根因

所有表面问题（巡航缺失、触发慢、CAPTURE失败）都指向同一根因：海康SDK的PTZ接口是异步的，发出指令后立即返回，但代码未等待球机物理到位。`_frame_settled()` 比较的是球机未动前的两帧旧图，帧差为0，被误判为"已稳定"。

日志证据：
- `PATROL_GOTO preset 1 → PATROL_DWELL settled` 间隔0秒
- SCAN模式4个预设位3秒内全扫完（物理上需要4-8秒/位）
- `SCAN_CAPTURE` 成功率≈0%（135次失败/15次进入识别）

## 包含任务

### OPT2-P0-1: PTZ指令后强制最小等待延迟

**修改文件**: `src/scheduler/state_machine.py`, `src/video/rtsp_source.py`, `src/config.py`, `config/config.yaml`

- 在 `_frame_settled()` 中增加最小等待保护：PTZ指令后至少等待 `min_wait_after_cmd`(0.8s) 或 `min_wait_after_zoom`(1.0s) 才开始稳图检测
- 等待期结束后调用 `video.flush()` 清空RTSP缓冲区旧帧
- 新增 `_record_ptz_cmd()` 方法统一管理PTZ命令时间戳和稳图状态重置
- 所有PTZ命令调用点（`_handle_patrol_goto`, `_handle_scan_goto_preset`, `_handle_scan_pick`, `_handle_scan_zoom_out`）统一使用 `_record_ptz_cmd()`
- `RtspSource` 新增 `flush(n_frames=5)` 方法

### OPT2-P0-2: 稳图判定改为两阶段（先变化再稳定）

**修改文件**: `src/scheduler/state_machine.py`, `src/config.py`, `config/config.yaml`

- `_frame_settled()` 从单阶段改为状态机：
  - `WAITING_MOTION`: 帧差 > motion_th(15.0) 时转入 MOTION_DETECTED
  - `MOTION_DETECTED`: 帧差 < settle_diff_th(5.0) 且连续 stable_frames(3) 帧时判定为稳定
- 必须先检测到运动（证明球机在动），再检测到稳定（证明球机停了）
- `settle_diff_th` 从 8.0 收紧到 5.0

### OPT2-P0-3: PTZ指令防抖与去重

**修改文件**: `src/sdk/hik_ptz.py`, `src/main.py`

- `HikPTZ` 类新增 `_last_cmd_ts` 和 `min_interval` 属性
- `goto_preset()` 和 `zoom_to_bbox()` 调用前检查最小间隔，不足则 sleep 补齐
- 防止SCAN模式下短时间内连续发出多个goto_preset指令导致球机指令队列拥塞
- `src/main.py` 从配置传入 `min_interval=cfg.ptz.min_wait_after_cmd`

## 新增配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `ptz.min_wait_after_cmd` | 0.8 | goto_preset后最小等待(秒) |
| `ptz.min_wait_after_zoom` | 1.0 | zoom_to_bbox后最小等待(秒) |
| `ptz.motion_th` | 15.0 | 帧差>此值视为球机在运动 |
| `ptz.stable_frames` | 3 | 连续N帧低于settle_diff_th才算稳定 |
| `ptz.settle_diff_th` | 5.0 | 稳定阈值（从8.0收紧） |

## 预期效果

- 巡航缺失问题消失：每个preset之间有充足时间让球机物理转动
- SCAN_CAPTURE成功率从0%提升到>50%：zoom后真正等待球机到位
- 日志中PATROL_GOTO到settled间隔≥0.8秒且反映真实转动时间
- 4个preset扫完耗时≥16秒（而非之前的3秒）
