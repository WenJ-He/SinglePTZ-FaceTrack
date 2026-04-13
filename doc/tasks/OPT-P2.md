### 任务完成报告

#### 1. 任务信息
- 任务ID：OPT-P2
- 任务名称：P2 参数批量调优

#### 2. 实现内容

**配置参数调整**

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| capture.timeout | 4.0 | 3.0 | 人体定位更准后CAPTURE不需要那么久 |
| capture.tracking.safe_zone_ratio | 0.6 | 0.75 | 减少不必要的追踪修正 |
| patrol.min_confirm_frames | 2 | 3 | 人体框更稳定，多确认一帧防误触 |
| quality blur_th | 80.0(硬编码) | 50.0(可配置) | H.264压缩流拉普拉斯方差普遍偏低 |

**代码改动**
- `quality.py`: blur_th 默认值 80→50
- `capture_tracker.py`: cfg类型从 CaptureTrackingConfig 改为 CaptureConfig，quality_ok 调用传入 blur_th
- `capture_tracker.py` 所有 tracking 子字段改为 self.cfg.tracking.xxx
- `state_machine.py`: CaptureTracker 构造传入 self.cfg.capture（完整CaptureConfig）
- `config.py` + `config.yaml`: 新增 capture.quality_blur_th 字段

#### 3. 修改文件
- `config/config.yaml`
- `src/config.py`
- `src/utils/quality.py`
- `src/scheduler/capture_tracker.py`
- `src/scheduler/state_machine.py`

#### 4. 测试情况
- 测试方式：Python语法检查 + 配置加载验证 + quality_ok 功能测试
- 测试结果：所有文件语法正确，config加载正确，quality_ok 可配置参数正常工作

#### 5. 当前状态
- 是否完成：✅
