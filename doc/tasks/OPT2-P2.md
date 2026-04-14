# OPT2-P2: 性能与环境优化

**批次**: OPT2-P2 (P2 建议改)
**日期**: 2026-04-14
**状态**: 已完成

## 问题描述

非阻塞性问题，改了能让系统更快更稳定、调参有据可依。

## 包含任务

### OPT2-P2-2: PATROL和SCAN的稳图超时差异化

**修改文件**: `src/scheduler/state_machine.py`, `src/config.py`, `config/config.yaml`

- 跨预设位场景（PATROL_GOTO、SCAN_GOTO_PRESET）使用 `settle_timeout_long`(2.5s)，因为球机需要在大角度间转动
- 小位移场景（SCAN_PICK zoom、SCAN_ZOOM_OUT）使用 `settle_timeout_short`(1.0s)，因为3D定位或回原预设位的移动距离小
- 保留原 `settle_timeout`(2.0s) 作为默认值

### OPT2-P2-3: 增加PTZ状态反馈日志

**修改文件**: `src/scheduler/state_machine.py`

- 在关键节点增加统一格式的 timing 日志：`[TIMING] state=X, phase=Y, elapsed=Zms`
- 打点阶段：
  - `cmd_issued`: PTZ指令发出
  - `min_wait_done`: 最小等待期结束
  - `motion_detected`: 检测到球机开始运动
  - `settled`: 检测到球机运动停止（帧稳定）
- 使用 `_timing_start` 记录每次PTZ指令发出的时间，计算各阶段耗时

### OPT2-P2-1: 启用GPU推理

**状态**: 环境配置任务，需现场执行

诊断命令：
```bash
nvidia-smi  # 确认GPU可见
pip list | grep onnxruntime  # 应为onnxruntime-gpu
python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
```

## 新增配置参数

| 参数 | 值 | 说明 |
|------|------|------|
| `ptz.settle_timeout_long` | 2.5 | 跨预设位稳图超时(秒) |
| `ptz.settle_timeout_short` | 1.0 | 3D定位小位移稳图超时(秒) |

## 预期效果

- 跨预设位有充足时间稳图，不再因超时提前进入下一阶段
- 日志能清楚看到每个PTZ操作的各阶段耗时，便于后续调参
- 识别性能瓶颈有据可依
