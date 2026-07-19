# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a YOLO11-based obstacle and person detection + pathfinding pipeline targeting Raspberry Pi 5. The end-to-end flow is:

1. **Camera capture** → capture a frame from a CSI or USB camera
2. **YOLO inference** → detect obstacles/persons
3. **GSD grid projection** → project detection boxes to a world-coordinate occupancy grid (meters), inflate obstacles, auto-select a goal
4. **A\* pathfinding** → find an optimal, collision-safe path from the vehicle to the goal

There are **two generations** of this pipeline in the repo:

- **Legacy loose scripts** (`修改后的yolo`, `astar`) — the original single-frame prototype. `修改后的yolo` maps bbox pixels straight to grid cells via integer division (no world projection); `astar` runs over the resulting `grid.npy` with a hardcoded start/goal. Kept for reference.
- **New modular pipeline** (`deploy_raspberry_pi/src/`) — the maintained implementation. Proper GSD world-coordinate projection, obstacle inflation, automatic goal selection, and a refactored importable A\*. **Prefer this for new work.**

Detection can also be run in a standalone streaming loop (`deploy_raspberry_pi/src/detect_camera_cli.py`) for continuous real-time monitoring.

## Key Files

| File | Role |
|------|------|
| `deploy_raspberry_pi/src/detect_to_path.py` | **End-to-end pipeline**: capture/image → YOLO → GSD grid → A\*. CLI with `--demo` (synthetic, no model/camera), `--image`, `--picamera` modes |
| `deploy_raspberry_pi/src/grid_map.py` | `CameraModel` (GSD from drone height + FOV), `build_grid` (box→world→cell projection + inflation), `select_goal` (auto goal), `to_ascii` |
| `deploy_raspberry_pi/src/astar_planner.py` | Importable A\* (`astar`, `plan`, `nearest_free`); 4/8-connected with diagonal corner-cut guard |
| `deploy_raspberry_pi/src/onnx_tools.py` | Edge-inference acceleration: export `.pt`→ONNX/NCNN, INT8 dynamic quantization, FPS benchmark |
| `deploy_raspberry_pi/tests/test_pipeline.py` | Runnable tests for GSD math, inflation, goal selection, and A\* (numpy-only, no model/camera) |
| `deploy_raspberry_pi/src/detect_camera_cli.py` | `CameraDetector` class — continuous real-time detection loop with optional video save and FPS overlay |
| `修改后的yolo` | *(legacy)* Captures one frame, naive pixel→grid mapping, saves `grid.npy` |
| `astar` | *(legacy)* Loads `grid.npy`, runs A\* with hardcoded start/goal |
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

### End-to-end pipeline (detect → grid → A\*)

```bash
# Offline self-check — synthetic obstacles, NO model or camera needed
python3 deploy_raspberry_pi/src/detect_to_path.py --demo --diag

# Run on a still image (drone height 12 m, 66° HFOV, 0.2 m/cell grid)
python3 deploy_raspberry_pi/src/detect_to_path.py \
    --model deploy_raspberry_pi/best_person.pt --image frame.jpg \
    --height 12 --hfov 66 --cell 0.2 --inflate 0.3 --diag

# Live from CSI camera
python3 deploy_raspberry_pi/src/detect_to_path.py \
    --model deploy_raspberry_pi/best_person.pt --picamera --height 10 --hfov 66

# Run the test suite (numpy-only)
python3 deploy_raspberry_pi/tests/test_pipeline.py
```

### Edge-inference acceleration (target >20 FPS on Pi 5)

```bash
# Export to NCNN (usually fastest on Pi 5 ARM CPU — recommended)
python3 deploy_raspberry_pi/src/onnx_tools.py export \
    deploy_raspberry_pi/best_person.pt --format ncnn --imgsz 640

# Export to ONNX + INT8 dynamic quantization (no calibration set needed)
python3 deploy_raspberry_pi/src/onnx_tools.py export \
    deploy_raspberry_pi/best_person.pt --format onnx --int8

# Benchmark any weight (.pt / .onnx / ncnn dir) to verify the FPS target
python3 deploy_raspberry_pi/src/onnx_tools.py benchmark best_person_ncnn_model --imgsz 640
```

The exported `.onnx` / `*_ncnn_model` can be passed straight to `--model` in both
`detect_to_path.py` and `detect_camera_cli.py` — Ultralytics `YOLO()` loads them natively,
so no inference code changes are needed.

### Detection & legacy scripts

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

# Legacy single-frame prototype
python3 修改后的yolo   # naive grid.npy from one frame
python3 astar          # A* over grid.npy with hardcoded start/goal
```

## Architecture Details

### End-to-end pipeline (`detect_to_path.py`)
Ties the previously-isolated scripts into one closed loop:
`grab_frame → detect_boxes → build_grid (GSD) → select_goal → plan (A*)`.
Three input sources make it testable without hardware: `--demo` (synthetic boxes,
no model/camera), `--image` (still frame), `--picamera`/USB (live). Outputs `grid.npy`,
`path.npy`, and an ASCII map. The module is import-safe both as a package and when run
directly (dual `from .grid_map` / `from grid_map` import guard).

### GSD grid projection (`grid_map.py`)
This is the real implementation of research goal ③ (boxes → grid), replacing the legacy
naive pixel-division approach:
- `CameraModel.gsd()` computes meters-per-pixel from a nadir camera: ground footprint =
  `2 · height · tan(FOV/2)`, divided by pixel count. Square-pixel assumption unless `vfov_deg` is given.
- `pixel_to_world()` maps pixels to a world frame **centered on the drone nadir point**
  (x right, y forward; image-y is inverted so up-image = +forward).
- `build_grid()` projects each box's 4 corners to world→cell and fills the covered cell range,
  then inflates (`inflate_m`) so the vehicle keeps clearance.
- `inflate_obstacles()` is a **Manhattan-distance dilation** (iterated 4-neighbour OR-shift),
  matching the proposal's "曼哈顿距离膨胀".
- `select_goal()` scans the front region (small `row` = far/forward) for the window with the
  most free cells, tie-breaking toward forward + center — the proposal's auto-goal selection.

### A\* (`astar_planner.py`)
- 4-connected (Manhattan heuristic) or 8-connected (`allow_diag=True`, **octile** heuristic +
  diagonal cost ≈ 1.414). Diagonal moves are blocked when both orthogonal neighbours are
  obstacles (no corner-cutting through wall junctions).
- `nearest_free()` snaps start/goal off obstacles; `plan()` is the convenience entry that
  snaps both ends then runs A\*, returning `(path, snapped_start, snapped_goal)`.
- Grid convention: `grid[row, col]`, row 0 = forward/far, `start = (H-2, W//2)` (vehicle near
  bottom-center), goal from `select_goal()`.

### Edge-inference acceleration (`onnx_tools.py`)
Implements the proposal's deployment path (PyTorch → ONNX → quantization) to push past the
current ~12.6 FPS toward the >20 FPS target. `export_model()` wraps Ultralytics export to
ONNX or NCNN; `quantize_dynamic_onnx()` does INT8 dynamic quantization (no calibration set);
`benchmark()` measures FPS. Heavy deps (`ultralytics`, `onnxruntime`) are import-guarded so
the CLI loads on any machine.

### Camera abstraction (`detect_camera_cli.py`)
`CameraDetector.open_camera()` tries `picamera2` first; falls back to `cv2.VideoCapture(0)`. Both paths are normalised to BGR numpy arrays before inference. The return value is `(camera_handle, is_picamera: bool)` so the cleanup path knows which API to call.

### Legacy grid map (`修改后的yolo`) and A\* (`astar`)
- `修改后的yolo`: 1 cell = `GRID_SIZE×GRID_SIZE` pixels (default 20), boxes mapped by integer
  division (no world projection), obstacle classes via `OBSTACLE_CLASSES`, saved to `grid.npy`.
- `astar`: 4/8-connected A\* with `nearest_free()` snapping; start/goal edited in-source.
- **Superseded by `grid_map.py` + `astar_planner.py`** — use those for new work.

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
