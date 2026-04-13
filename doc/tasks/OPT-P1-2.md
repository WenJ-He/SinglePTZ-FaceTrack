### 任务完成报告

#### 1. 任务信息
- 任务ID：OPT-P1-2
- 任务名称：ISAPI高清抓图用于CAPTURE

#### 2. 实现内容
- 新增 `src/sdk/hik_isapi.py`：HikISAPI 类
  - HTTP Digest Auth 认证
  - `capture_jpeg()` 调用 ISAPI /picture 接口获取硬件编码JPEG
  - 超时3秒，失败时返回None
- 修改 `capture_tracker.py`：
  - 构造函数新增 `isapi` 参数
  - collect 动作时：优先用 ISAPI 抓图做 face crop，ISAPI 帧上重新检测定位
  - ISAPI 不可用时 fallback 到 RTSP 帧
- 修改 `main.py`：根据 `hik.isapi_enabled` 创建 HikISAPI 实例传入 scheduler
- 修改 `state_machine.py`：构造函数新增 `isapi` 参数，传给 CaptureTracker
- 新增配置 `hik.isapi_enabled: true`

#### 3. 修改/创建文件
- `src/sdk/hik_isapi.py` — 新建
- `src/scheduler/capture_tracker.py`
- `src/scheduler/state_machine.py`
- `src/main.py`
- `src/config.py`
- `config/config.yaml`
- `requirements.txt` — 新增 requests>=2.28

#### 4. 测试情况
- 测试方式：Python语法检查 + 配置加载验证 + import 验证
- 测试结果：所有文件语法正确，config 加载正确，import 链路完整

#### 5. 当前状态
- 是否完成：✅
