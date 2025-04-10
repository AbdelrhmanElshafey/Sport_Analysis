# 🏟️ Sport Analysis

A Python-based AI pipeline for in‑depth soccer match analysis. This repo contains two main modules:

> **Note:** This repository includes implementation code only; trained model weights are **not** included.
1. **Heatmap & Radar & Space Control Generation** (`soccer_code_1`)  
2. **Detection & Analytics** (`soccer_code_2`)  

---

## 🧠 Methodology

### Heatmap & Radar (`soccer_code_1`)
- **Data Buffering**: Aggregate each team’s transformed pitch coordinates over time.
- **Radar Overlay**:  
  - `render_radar()` builds and overlays a live “radar” of cumulative movement.
- **Heatmap Generation**:  
  - `run_heatmap_team0()` implements first team heatmap as video.
  - `run_heatmap_team1()` implements second team heatmap as video.  
- **Space Control**: 
  - `run_voronoi()` display space controlled by each team.

### Detection & Analytics (`soccer_code_2`)
- **Object Detection**: YOLO‑style models for players, ball, and pitch.  
- **Tracking**: ByteTrack for consistent IDs across frames.  
- **Speed & Distance**: Compute displacement per frame → instantaneous speed & total distance.  
- **Ball Control**: Track possession duration per player.

---
