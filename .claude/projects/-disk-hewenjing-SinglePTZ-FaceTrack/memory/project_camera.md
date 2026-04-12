---
name: Camera hardware specs
description: Hikvision iDS-2DE2402IX-D3/W/XM PTZ camera specs and constraints - only 2x optical zoom
type: project
---

Camera: iDS-2DE2402IX-D3/W/XM (4MP, 2x optical zoom 2.8-6mm, indoor ceiling-mount mini PTZ dome)
- FOV: 100° wide to 45° tele, Pan 350° (10° dead zone), Tilt 0-90°
- Connection: 192.168.0.249:8000, admin, channel=1
- 3D positioning (PTZSelZoomIn_EX) confirmed supported
- 300 presets, 8 cruise routes

**Why:** 2x zoom is the biggest constraint — can't do dramatic close-ups from far away. Design relies on 4-preset coverage (4×100° = 400° > 350° with overlap) + 4MP resolution + PTZ pan-to-center rather than high-zoom magnification.

**How to apply:** All PTZ timing parameters (settle_timeout etc.) should be shorter than typical 20x/30x ball machines. Expect face pixels ~60-80px at 3m wide-angle, ~120-160px after 2x zoom.
