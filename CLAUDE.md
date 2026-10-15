# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a YOLO11-based obstacle and person detection + pathfinding pipeline targeting Raspberry Pi 5. The end-to-end flow is:

1. **Camera capture** → capture a frame from a CSI or USB camera
2. **YOLO inference** (`修改后的yolo`) → detect obstacles/persons, convert bounding boxes to a binary grid map, save as `grid.npy`
3. **A\* pathfinding** (`astar`) → load `grid.npy`, find optimal path from a start cell to a goal cell

Detection can also be run in a standalone streaming loop (`deploy_raspberry_pi/src/detect_camera_cli.py`) for continuous real-time monitoring.

The algorithms live in the `deploy_raspberry_pi/rescue/` package (numpy-only except `comm`), mapped 1:1 to the project application (申报书) research items — see `docs/结项成果说明.md`. `修改后的yolo` and `astar` are now thin compatibility wrappers around `drone_node.py` and `rescue.planner`. No drone/car hardware exists yet; `rescue/sim.py` simulates the full air-ground loop.

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
| `deploy_raspberry_pi/rescue/geo_mapping.py` | Camera model, GSD, pixel↔ground projection with IMU attitude, grid rasterisation, Manhattan inflation, goal selection |
| `deploy_raspberry_pi/rescue/planner.py` | A* (octile heuristic, no corner cutting), path simplification |
| `deploy_raspberry_pi/rescue/lidar_gap.py` | 2D-lidar d(θ) profile, gap detection, corridor-checked desired heading |
| `deploy_raspberry_pi/rescue/local_map.py` | `LidarMapper`: lidar hits mark obstacles, ray carving clears false aerial obstacles; shared by `car_node.py` and `sim.py` |
| `deploy_raspberry_pi/rescue/control.py` | Pure pursuit, PID heading control, differential drive, recovery, GPIO motor driver |
| `deploy_raspberry_pi/rescue/person_locate.py` | Person distance/bearing from car camera intrinsics + mount height; `PersonTracker` multi-frame confirmation |
| `deploy_raspberry_pi/rescue/comm.py` | MQTT (paho 1.x/2.x) + loopback link; compressed grid-map messages |
| `deploy_raspberry_pi/rescue/sim.py` | Full air-ground simulation and Monte Carlo comparison; `run_mission(on_step=...)` callback for recording; `--recall/--car-recall` set detector recall |
| `deploy_raspberry_pi/rescue/viz.py` | Per-module figures (mapping, astar, gap, person, control, comm, baro, demo, frames) and mission animation (mp4/gif); `--report` renders compact print versions for the Word report |
| `deploy_raspberry_pi/drone_node.py` / `car_node.py` | Air-side and ground-side entry points; `car_node` also runs lidar map correction + replanning and optional car-camera person detection (`--person-model`) |
| `deploy_raspberry_pi/rescue/baro.py` / `pose_publisher.py` | BMP388 I2C driver (Bosch float compensation) + relative altimeter; pose node publishing `{altitude, pitch, roll, yaw}` to MQTT `rescue/pose` (IMU attitude from a ROS 2 `sensor_msgs/Imu` topic) for `drone_node.py --pose-topic` |
| `deploy_raspberry_pi/benchmark.py` | ONNX/NCNN export, INT8, FPS comparison, long-run soak test |
| `deploy_raspberry_pi/yolo_examples.py` | Runs both models on given images and composes the (a) collapsed / (b) person example figure |

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

# Low-resource / headless mode: 320 input, detect every other frame, no window
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt \
    --imgsz 320 --skip-frames 1 --headless --no-fps

# Generate grid map from one captured frame
python3 修改后的yolo   # produces grid.npy in the working directory

# Run A* over the generated grid
python3 astar          # reads grid.npy, prints path, shows ASCII visualisation
```

## Tests and simulation

```bash
cd deploy_raspberry_pi
python3 -m pytest tests -q                                        # numpy-only unit tests
python3 -m rescue.sim --seed 1 --figure sim.png                   # single run + figure
python3 -m rescue.sim --trials 30 --seed 100 --json results.json  # Monte Carlo (writes docs data)
python3 -m rescue.sim --trials 30 --seed 100 --recall 0.25 --car-recall 0.25  # sensitivity to the measured model recall
python3 -m rescue.viz all --out viz                                # per-module figures + mission animations (needs matplotlib, ffmpeg for mp4)
```

## Architecture Details

### Camera abstraction (`detect_camera_cli.py`)
`CameraDetector.open_camera()` tries `picamera2` first; falls back to `cv2.VideoCapture(0)`. Both paths are normalised to BGR numpy arrays before inference. The return value is `(camera_handle, is_picamera: bool)` so the cleanup path knows which API to call.

### Grid map (`rescue/geo_mapping.py`, via `drone_node.py` / `修改后的yolo`)
- Ground frame: origin below the drone, +x = image right, +y = image up; `grid[row, col]` with row 0 = top
- GSD from camera hfov + altitude; box corners are ray-cast to the ground using pitch/roll/yaw
- Resolution in metres per cell (default 0.25); obstacles inflated by Manhattan radius (`--inflate`)
- Model classes: `best_collapsed.pt` → `collapsed` (obstacle); `best_person.pt` → `person` (target, never an obstacle)
- Outputs `grid.npy`, `grid_meta.json` (georeference + goal), `grid_map.png`

### A\* (`rescue/planner.py`, via `astar`)
- 4-connected (Manhattan) or 8-connected (octile heuristic, diagonal cost √2, corner cutting forbidden)
- `nearest_free()` snaps start/goal onto free cells
- `astar` wrapper: start = bottom-centre; goal from `grid_meta.json` if present, else top-centre

## Models

Two `.pt` files are included in `deploy_raspberry_pi/`:

- `best_person.pt` — person detection
- `best_collapsed.pt` — collapsed-structure / debris detection

To switch models, pass the path via `--model` (`drone_node.py` accepts several). `修改后的yolo` runs both bundled models by default.

## Dependencies

CPU-only PyTorch is used throughout (no CUDA). Key versions:

- `torch==2.0.1` (cpu wheel)
- `torchvision==0.15.2` (cpu wheel)
- `ultralytics==8.3.0`
- `opencv-python==4.8.1.78`
- `numpy>=1.24.0`

`picamera2` is installed as a system package (`sudo apt install python3-picamera2`) and accessed via `--system-site-packages` in the venv.
