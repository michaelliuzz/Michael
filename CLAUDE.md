# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a YOLO11-based obstacle and person detection + pathfinding pipeline targeting Raspberry Pi 5. The end-to-end flow is:

1. **Camera capture** → capture a frame from a CSI or USB camera
2. **YOLO inference** (`修改后的yolo`) → detect obstacles/persons, convert bounding boxes to a binary grid map, save as `grid.npy`
3. **A\* pathfinding** (`astar`) → load `grid.npy`, find optimal path from a start cell to a goal cell

Detection can also be run in a standalone streaming loop (`deploy_raspberry_pi/src/detect_camera_cli.py`) for continuous real-time monitoring.

## Key Files

| File | Role |
|------|------|
| `修改后的yolo` | Captures one frame, runs YOLO inference, builds `grid.npy` (0=free, 1=obstacle) |
| `astar` | Loads `grid.npy`, runs A\* (4- or 8-connected), prints/visualises path |
| `deploy_raspberry_pi/src/detect_camera_cli.py` | `CameraDetector` class — continuous real-time detection loop with optional video save and FPS overlay |
| `deploy_raspberry_pi/best_collapsed.pt` | YOLO11 model for collapsed-structure detection |
| `deploy_raspberry_pi/best_person.pt` | YOLO11 model for person detection |
| `deploy_raspberry_pi/requirements_raspberry_pi.txt` | Pinned CPU-only PyTorch + Ultralytics deps |
| `deploy_raspberry_pi/raspberry_pi_setup_venv.sh` | One-shot venv setup (installs system pkgs + Python deps via Tsinghua/Aliyun mirrors) |
| `deploy_raspberry_pi/start_venv.sh` | Activates venv and launches detection |
| `deploy_raspberry_pi/autostart_detection.sh` | Crontab `@reboot` helper; set `PROJECT_DIR` before use |

## Setup (Raspberry Pi)

```bash
# 1. Install system deps and create venv (runs once)
bash deploy_raspberry_pi/raspberry_pi_setup_venv.sh

# 2. Activate venv for every subsequent session
source deploy_raspberry_pi/venv/bin/activate

# Or use the wrapper:
bash deploy_raspberry_pi/start_venv.sh
```

The setup script creates the venv with `--system-site-packages` so that `picamera2` (installed as a system package) is visible inside the venv.

## Common Commands

```bash
# Real-time detection (CSI camera)
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt --picamera

# Real-time detection (USB camera)
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt

# Save video, run for 60 s, no FPS overlay (reduces CPU load)
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt \
    --save --duration 60 --no-fps

# Low-resource mode (skip-frames must be added to CLI if needed)
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt \
    --no-fps --conf 0.3

# Generate grid map from one captured frame
python3 修改后的yolo   # produces grid.npy in the working directory

# Run A* over the generated grid
python3 astar          # reads grid.npy, prints path, shows ASCII visualisation
```

## Architecture Details

### Camera abstraction (`detect_camera_cli.py`)
`CameraDetector.open_camera()` tries `picamera2` first; falls back to `cv2.VideoCapture(0)`. Both paths are normalised to BGR numpy arrays before inference. The return value is `(camera_handle, is_picamera: bool)` so the cleanup path knows which API to call.

### Grid map (`修改后的yolo`)
- Resolution: 1 grid cell = `GRID_SIZE × GRID_SIZE` pixels (default 20 px)
- Obstacle classes are configurable via `OBSTACLE_CLASSES`
- Bounding boxes are mapped to grid indices with integer division; all cells inside a box are set to 1
- Grid is saved with `np.save("grid.npy", grid)`

### A\* (`astar`)
- Supports 4-connected (Manhattan heuristic) and 8-connected (`allow_diag=True`, diagonal cost ≈ 1.414)
- `nearest_free()` auto-snaps start/goal to a free cell if they fall on an obstacle
- Default: start = bottom-centre row, goal = top-centre row (modify `start`/`goal` variables directly)

## Models

Two `.pt` files are included in `deploy_raspberry_pi/`:

- `best_person.pt` — person detection
- `best_collapsed.pt` — collapsed-structure / debris detection

To switch models, pass the path via `--model`. The `修改后的yolo` script uses `MODEL_PATH = "best.pt"` — update that variable to point to one of the bundled models.

## Dependencies

CPU-only PyTorch is used throughout (no CUDA). Key versions:

- `torch==2.0.1` (cpu wheel)
- `torchvision==0.15.2` (cpu wheel)
- `ultralytics==8.3.0`
- `opencv-python==4.8.1.78`
- `numpy>=1.24.0`

`picamera2` is installed as a system package (`sudo apt install python3-picamera2`) and accessed via `--system-site-packages` in the venv.
