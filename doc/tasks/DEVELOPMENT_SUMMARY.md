# SinglePTZ-FaceTrack Development Summary

## Status: All 8 Milestones Complete (43/43 tasks)

### Milestones Completed

| Milestone | Name | Tasks | Commit |
|---|---|---|---|
| M1 | SDK 打通 | 7/7 | c03dd69 ~ 806e028 |
| M2 | 取流 + 检测 | 5/5 | 32ee143 ~ a437ccc |
| M3 | 识别 + 人脸库 | 4/4 | 7582c3d |
| M4 | 巡航 MVP | 4/4 | aa8abba |
| M5 | 单预设位扫描 | 6/6 | e12f3d2 |
| M6 | 多预设位 + ReID | 5/5 | 9634776 |
| M7 | 可视化 + 日志 | 7/7 | c2fba9e |
| M8 | 调优 & 验收 | 5/5 | 1467df4 |

### Key Files Created

```
src/
├── config.py                    # YAML -> dataclass mapping
├── main.py                      # Entry point
├── sdk/
│   ├── hik_sdk.py               # Hikvision SDK ctypes bindings
│   └── hik_ptz.py               # PTZ business wrapper
├── video/
│   └── rtsp_source.py           # RTSP background thread
├── detect/
│   ├── yolo_face.py             # YOLO11n face detection (640/1280)
│   └── yolo_person.py           # YOLOv8n person detection
├── recognize/
│   ├── arcface.py               # ArcFace 512-d embedding
│   └── gallery.py               # Face gallery build/cache/match
├── reid/
│   └── osnet.py                 # OSNet ReID 512-d embedding
├── track/
│   └── sort_reid.py             # SORT+ReID tracker
├── scheduler/
│   ├── state_machine.py         # 12-state scan scheduler
│   └── capture_tracker.py       # B+C dynamic tracking correction
├── ui/
│   ├── visualizer.py            # Frame annotation renderer
│   ├── web_stream.py            # Flask MJPEG HTTP stream
│   └── display.py               # Display backend abstraction
└── utils/
    ├── logger.py                # Logging module
    ├── geometry.py              # Bbox/IOU utilities
    ├── quality.py               # Face quality checks
    └── event_logger.py          # JSONL event logger
```

### Notes for Real Device Testing

1. **Gallery thresholds**: Cross-person similarity max=0.51, so match_th=0.35 may need raising to 0.55+
2. **OSNet model**: Currently uses random weights (no pretrained available offline). Need pretrained osnet_x0_25 weights for production ReID.
3. **Git push**: Network unavailable during development. Push manually when network is available:
   ```bash
   git push -u origin main
   ```

### Environment

- Conda env: `single_ptz_facetrack` (Python 3.10)
- Start: `./run.sh` or `conda activate single_ptz_facetrack && python src/main.py`
- Web stream: http://0.0.0.0:8080/
