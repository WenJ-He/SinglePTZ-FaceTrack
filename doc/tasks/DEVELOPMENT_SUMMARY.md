# SinglePTZ-FaceTrack 开发总结

## 状态：全部8个里程碑完成 + 实测优化完成（43/43 + 7/7 OPT）

### 已完成里程碑

| 里程碑 | 名称 | 任务数 | 提交 |
|---|---|---|---|
| M1 | SDK 打通 | 7/7 | c03dd69 ~ 806e028 |
| M2 | 取流 + 检测 | 5/5 | 32ee143 ~ a437ccc |
| M3 | 识别 + 人脸库 | 4/4 | 7582c3d |
| M4 | 巡航 MVP | 4/4 | aa8abba |
| M5 | 单预设位扫描 | 6/6 | e12f3d2 |
| M6 | 多预设位 + ReID | 5/5 | 9634776 |
| M7 | 可视化 + 日志 | 7/7 | c2fba9e |
| M8 | 调优 & 验收 | 5/5 | 1467df4 |

### 创建的关键文件

```
src/
├── config.py                    # YAML → dataclass 配置映射
├── main.py                      # 程序入口
├── sdk/
│   ├── hik_sdk.py               # 海康SDK ctypes绑定
│   ├── hik_ptz.py               # PTZ业务封装
│   └── hik_isapi.py             # ISAPI高清抓图接口（OPT-P1-2新增）
├── video/
│   └── rtsp_source.py           # RTSP后台取流
├── detect/
│   ├── yolo_face.py             # YOLO11n 人脸检测（640/1280）
│   └── yolo_person.py           # YOLOv8n 人体检测
├── recognize/
│   ├── arcface.py               # ArcFace 512-d 嵌入
│   └── gallery.py               # 人脸库构建/缓存/匹配
├── reid/
│   └── osnet.py                 # OSNet ReID 512-d 嵌入
├── track/
│   └── sort_reid.py             # SORT+ReID 跟踪器
├── scheduler/
│   ├── state_machine.py         # 12态扫描调度器（含聚焦判定）
│   └── capture_tracker.py       # B+C动态追踪修正（含ISAPI抓图）
├── ui/
│   ├── visualizer.py            # 帧标注渲染
│   ├── web_stream.py            # Flask MJPEG HTTP流
│   └── display.py               # 显示后端抽象
└── utils/
    ├── logger.py                # 日志模块
    ├── geometry.py              # Bbox/IoU工具函数
    ├── quality.py               # 人脸质量检查
    └── event_logger.py          # JSONL事件日志
```

### 实测优化（OPT）

| 批次 | 任务 | 改动 | 提交 |
|---|---|---|---|
| OPT-P0 | 巡航改用人体检测 + Web画质 + 稳图阈值 | config/state_machine | b044892 |
| OPT-P1-1 | 聚焦完成判定（拉普拉斯方差） | state_machine | ad4825e |
| OPT-P1-2 | ISAPI高清抓图用于CAPTURE | hik_isapi(新)/capture_tracker/main | a1f8a8d |
| OPT-P2 | 参数批量调优 | config/quality/capture_tracker | 6ac8061 |

### 实机测试注意事项

1. **人脸库阈值**：跨人相似度最大值=0.51，match_th=0.35 可能需要上调至 0.55+
2. **OSNet模型**：当前使用随机权重（离线环境无法下载预训练），需替换为预训练 osnet_x0_25 权重
3. **Git推送**：开发期间网络不可用，需手动推送：
   ```bash
   git push -u origin main
   ```

### 运行环境

- Conda 环境：`single_ptz_facetrack`（Python 3.10）
- 启动方式：`./run.sh` 或 `conda activate single_ptz_facetrack && python src/main.py`
- Web流地址：http://0.0.0.0:8080/
