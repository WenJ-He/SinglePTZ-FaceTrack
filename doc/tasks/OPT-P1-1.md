### 任务完成报告

#### 1. 任务信息
- 任务ID：OPT-P1-1
- 任务名称：增加聚焦完成判定

#### 2. 实现内容
- 在 `state_machine.py` 新增 `_focus_settled(frame)` 方法
  - 维护最近3帧的拉普拉斯方差历史
  - 连续3帧方差递增且最新值 > focus_min_laplacian 时判定聚焦完成
- 修改 `_handle_scan_settle()`：
  - 帧差稳图通过后，追加 `_focus_settled()` 检查
  - 两项都通过才进入 CAPTURE
  - 超时时仍继续（降级处理，不卡住状态机）
- 新增配置项 `ptz.focus_min_laplacian`（默认50.0）

#### 3. 修改/创建文件
- `src/scheduler/state_machine.py` — 新增 `_focus_settled` 方法，修改 `_handle_scan_settle`
- `src/config.py` — PtzConfig 新增 `focus_min_laplacian` 字段
- `config/config.yaml` — 新增 `focus_min_laplacian: 50.0`

#### 4. 测试情况
- 测试方式：Python语法检查 + 配置加载验证
- 测试结果：config 加载正确，state_machine.py 语法检查通过

#### 5. 当前状态
- 是否完成：✅
