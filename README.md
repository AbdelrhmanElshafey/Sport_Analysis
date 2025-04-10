# 🏟️ Sport Analysis

A Python-based AI pipeline for in‑depth soccer match analysis. This repo contains two main modules:

> **Note:** This repository includes implementation code only; pre-trained model weights are **not** included.
1. **Heatmap & Radar Generation** (`soccer_code_1`)  
2. **Detection & Analytics** (`soccer_code_2`)  

---

## 🧠 Methodology

### Heatmap & Radar (`soccer_code_1`)
- **Data Buffering**: Aggregate each team’s transformed pitch coordinates over time.  
- **Heatmap Generation**:  
  - `draw_single_team_heatmap()` writes per‑team density maps as video.  
- **Radar Overlay**:  
  - `render_radar()` builds and overlays a live “radar” of cumulative movement.

### Detection & Analytics (`soccer_code_2`)
- **Object Detection**: YOLO‑style models for players, ball, and pitch.  
- **Tracking**: ByteTrack for consistent IDs across frames.  
- **Speed & Distance**: Compute displacement per frame → instantaneous speed & total distance.  
- **Ball Control**: Track possession duration per player.

---
