"""
研究内容(3)：将 YOLO11 检测结果（像素框）转化为带真实尺度的二维栅格地图

流程（与申报书一致）:
    1. 由相机视场角 FOV 与图像分辨率求焦距，由无人机高度求每个像素对应的地面尺寸（GSD）
    2. 结合 IMU 姿态（俯仰/横滚/偏航）把像素射线投影到地平面，得到地面坐标
    3. 对每个障碍物框计算四个角的地面坐标，栅格化为若干栅格单元
    4. 对障碍格做一次曼哈顿距离膨胀（车体半宽 + 安全冗余）
    5. 在前方区域扫描“障碍最少、空白最多”的块作为终点；若空中发现人员，则直接以人员位置为终点

地面坐标系约定（单位：米）:
    原点 = 无人机正下方地面点
    +x  = 图像右方向，+y = 图像上方向（无人机机头方向），+z = 竖直向上
栅格约定:
    grid[row, col]，row=0 对应地图最上方（y 最大），与图像方向一致；0=可通行，1=障碍
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 1. 相机模型与 GSD
# ---------------------------------------------------------------------------

@dataclass
class CameraModel:
    """针孔相机模型，默认 640x480、水平视场角 62.2°（树莓派 Camera v2）。
    镜头畸变明显时需先标定内参并去畸变，再把标定得到的等效视场角填入 hfov_deg"""
    width: int = 640
    height: int = 480
    hfov_deg: float = 62.2       # 水平视场角（默认值为树莓派 Camera v2 的 62.2°，请按所用镜头的标称值或标定值修改）
    vfov_deg: float | None = None  # 为 None 时按方形像素由 hfov 推出

    @property
    def fx(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        if self.vfov_deg is None:
            return self.fx
        return (self.height / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2


def ground_sample_distance(cam: CameraModel, altitude_m: float) -> float:
    """竖直向下拍摄时，图像中心处一个像素对应的地面边长（米/像素）"""
    return altitude_m / cam.fx


@dataclass
class DronePose:
    """无人机位姿。角度单位为度，来源于九轴 IMU（姿态）与 BMP388 气压计（高度）"""
    altitude_m: float = 10.0
    pitch_deg: float = 0.0   # >0：相机向机头方向（+y）倾斜
    roll_deg: float = 0.0    # >0：相机向右侧（+x）倾斜
    yaw_deg: float = 0.0     # 地图相对图像的旋转，通常取 0 让地图与图像对齐


def altitude_from_pressure(pressure_hpa: float, ground_pressure_hpa: float,
                           temperature_c: float = 15.0) -> float:
    """气压测高（等温近似）：起飞前记录地面气压，飞行中用当前气压求相对高度"""
    t_k = temperature_c + 273.15
    return 29.271 * t_k * math.log(ground_pressure_hpa / pressure_hpa)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# 相机系（x 右、y 下、z 光轴）-> 地面系，相机竖直向下时
_R_NADIR = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])


def camera_to_ground_rotation(pose: DronePose) -> np.ndarray:
    return (_rot_z(math.radians(pose.yaw_deg))
            @ _rot_x(math.radians(pose.pitch_deg))
            @ _rot_y(-math.radians(pose.roll_deg))
            @ _R_NADIR)


def pixels_to_ground(uv: np.ndarray, cam: CameraModel, pose: DronePose) -> np.ndarray:
    """
    像素坐标 -> 地面坐标（米）

    Args:
        uv: (N, 2) 像素坐标 (u 向右, v 向下)
    Returns:
        (N, 2) 地面坐标 (x, y)；射线不与地面相交（指向地平线以上）的点为 NaN
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    rays_c = np.stack([(uv[:, 0] - cam.cx) / cam.fx,
                       (uv[:, 1] - cam.cy) / cam.fy,
                       np.ones(len(uv))], axis=1)
    rays_g = rays_c @ camera_to_ground_rotation(pose).T
    out = np.full((len(uv), 2), np.nan)
    hit = rays_g[:, 2] < -1e-6
    t = -pose.altitude_m / rays_g[hit, 2]
    out[hit, 0] = t * rays_g[hit, 0]
    out[hit, 1] = t * rays_g[hit, 1]
    return out


def ground_to_pixels(xy: np.ndarray, cam: CameraModel, pose: DronePose) -> np.ndarray:
    """地面坐标 -> 像素坐标（pixels_to_ground 的逆过程，用于仿真与校验）"""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    pts = np.column_stack([xy, np.zeros(len(xy))]) - np.array([0, 0, pose.altitude_m])
    pc = pts @ camera_to_ground_rotation(pose)  # 旋转矩阵正交，转置即逆
    u = cam.fx * pc[:, 0] / pc[:, 2] + cam.cx
    v = cam.fy * pc[:, 1] / pc[:, 2] + cam.cy
    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# 2. 栅格地图
# ---------------------------------------------------------------------------

@dataclass
class GridMap:
    """带地理参考的栅格地图"""
    grid: np.ndarray                 # uint8, 0=free 1=obstacle
    resolution: float                # 米/格
    x_min: float                     # 地图左边界（米）
    y_max: float                     # 地图上边界（米）
    raw: np.ndarray | None = field(default=None, repr=False)  # 膨胀前的障碍层

    @property
    def shape(self):
        return self.grid.shape

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((self.y_max - y) / self.resolution)), \
               int(math.floor((x - self.x_min) / self.resolution))

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """返回栅格中心的地面坐标"""
        return (self.x_min + (col + 0.5) * self.resolution,
                self.y_max - (row + 0.5) * self.resolution)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def to_dict(self) -> dict:
        return {"resolution": self.resolution, "x_min": self.x_min, "y_max": self.y_max,
                "shape": list(self.grid.shape)}


def image_footprint(cam: CameraModel, pose: DronePose) -> np.ndarray:
    """图像四角在地面上的投影（顺序：左上、右上、右下、左下）"""
    corners = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]], float)
    return pixels_to_ground(corners, cam, pose)


def empty_map_for_footprint(cam: CameraModel, pose: DronePose, resolution: float,
                            max_extent_m: float = 200.0) -> GridMap:
    """按图像在地面的覆盖范围建立空白栅格地图（倾斜过大时用 max_extent_m 截断）"""
    fp = image_footprint(cam, pose)
    fp = fp[~np.isnan(fp).any(axis=1)]
    if len(fp) < 2:
        raise ValueError("相机倾角过大，图像大部分位于地平线以上，无法建立地图")
    fp = np.clip(fp, -max_extent_m, max_extent_m)
    x_min, x_max = fp[:, 0].min(), fp[:, 0].max()
    y_min, y_max = fp[:, 1].min(), fp[:, 1].max()
    rows = max(1, int(math.ceil((y_max - y_min) / resolution)))
    cols = max(1, int(math.ceil((x_max - x_min) / resolution)))
    return GridMap(np.zeros((rows, cols), np.uint8), resolution, float(x_min), float(y_max))


def fill_polygon(gm: GridMap, poly_xy: np.ndarray, value: int = 1) -> None:
    """把地面多边形（凸四边形）覆盖到的栅格标记为 value（按栅格中心判定，并保证顶点所在格被标记）"""
    poly_xy = np.asarray(poly_xy, float)
    if np.isnan(poly_xy).any():
        return
    H, W = gm.grid.shape
    rc = np.array([gm.world_to_cell(x, y) for x, y in poly_xy])
    r0, r1 = max(0, rc[:, 0].min()), min(H - 1, rc[:, 0].max())
    c0, c1 = max(0, rc[:, 1].min()), min(W - 1, rc[:, 1].max())
    if r0 > r1 or c0 > c1:
        return
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    px = gm.x_min + (cc + 0.5) * gm.resolution
    py = gm.y_max - (rr + 0.5) * gm.resolution
    inside_pos = np.ones(px.shape, bool)
    inside_neg = np.ones(px.shape, bool)
    n = len(poly_xy)
    for i in range(n):
        ax, ay = poly_xy[i]
        bx, by = poly_xy[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside_pos &= cross >= 0
        inside_neg &= cross <= 0
    mask = inside_pos | inside_neg
    gm.grid[rr[mask], cc[mask]] = value
    for r, c in rc:
        if gm.in_bounds(r, c):
            gm.grid[r, c] = value


def inflate_manhattan(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """曼哈顿距离膨胀：把与障碍格曼哈顿距离 <= radius 的格子也标记为障碍"""
    out = grid.astype(bool).copy()
    for _ in range(max(0, radius_cells)):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out.astype(np.uint8)


@dataclass
class Detection:
    """一个检测结果（像素框）"""
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]


def detections_from_ultralytics(result) -> list[Detection]:
    """把 ultralytics 的 Results 对象转换为 Detection 列表"""
    names = result.names
    dets = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        dets.append(Detection(names[cls_id], float(box.conf[0]),
                              tuple(float(v) for v in box.xyxy[0])))
    return dets


# 两个训练模型的类别名。人员（person）不是障碍物，而是搜救目标
DEFAULT_OBSTACLE_CLASSES = ("debris", "stone", "rock", "wood", "tree", "rubble",
                            "building", "collapsed", "obstacle")
DEFAULT_TARGET_CLASSES = ("person", "human", "people")


@dataclass
class MapBuildResult:
    gridmap: GridMap
    persons_xy: list[tuple[float, float, float]]   # (x, y, conf) 空中发现的人员地面坐标
    gsd: float


def build_grid_map(detections: list[Detection], cam: CameraModel, pose: DronePose,
                   resolution: float = 0.25, inflate_radius_m: float = 0.3,
                   obstacle_classes=DEFAULT_OBSTACLE_CLASSES,
                   target_classes=DEFAULT_TARGET_CLASSES,
                   unknown_as_obstacle: bool = True,
                   gridmap: GridMap | None = None) -> MapBuildResult:
    """
    由一帧航拍检测结果构建栅格地图

    Args:
        resolution: 栅格分辨率（米/格）
        inflate_radius_m: 膨胀半径，一般取 小车半宽 + 安全冗余
        unknown_as_obstacle: 不在两个类别表中的类别是否当作障碍（保守处理）
        gridmap: 传入已有地图则在其上叠加（多帧融合），否则按图像覆盖范围新建
    """
    gm = gridmap if gridmap is not None else empty_map_for_footprint(cam, pose, resolution)
    raw = gm.raw.copy() if gm.raw is not None else gm.grid.copy()
    gm_raw = GridMap(raw, gm.resolution, gm.x_min, gm.y_max)
    obstacle_classes = {c.lower() for c in obstacle_classes}
    target_classes = {c.lower() for c in target_classes}
    persons = []
    for d in detections:
        name = d.cls_name.lower()
        x1, y1, x2, y2 = d.xyxy
        if name in target_classes:
            # 以框中心作为人员位置（俯视时人体投影中心）
            gx, gy = pixels_to_ground(np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]), cam, pose)[0]
            if not np.isnan(gx):
                persons.append((float(gx), float(gy), d.conf))
            continue
        if name not in obstacle_classes and not unknown_as_obstacle:
            continue
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
        fill_polygon(gm_raw, pixels_to_ground(corners, cam, pose))
    r = int(math.ceil(inflate_radius_m / gm.resolution))
    inflated = inflate_manhattan(gm_raw.grid, r)
    # 人员不进入障碍层；若其所在格被膨胀覆盖，select_goal 会用 nearest_free 选最近的可通行格作终点
    out = GridMap(inflated, gm.resolution, gm.x_min, gm.y_max, raw=gm_raw.grid)
    return MapBuildResult(out, persons, ground_sample_distance(cam, pose.altitude_m))


# ---------------------------------------------------------------------------
# 3. 终点选择
# ---------------------------------------------------------------------------

def nearest_free(grid: np.ndarray, cell: tuple[int, int], max_radius: int = 50):
    """在 cell 附近（按切比雪夫半径逐圈）寻找最近的可通行格；找不到返回 None"""
    H, W = grid.shape
    r0, c0 = cell
    r0 = min(max(r0, 0), H - 1)
    c0 = min(max(c0, 0), W - 1)
    if grid[r0, c0] == 0:
        return (r0, c0)
    best, best_d = None, None
    for rad in range(1, max_radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in (-rad, rad) if abs(dr) != rad else range(-rad, rad + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < H and 0 <= c < W and grid[r, c] == 0:
                    d = dr * dr + dc * dc
                    if best_d is None or d < best_d:
                        best, best_d = (r, c), d
        if best is not None:
            return best
    return None


def select_goal_free_block(grid: np.ndarray, block: int = 8, front_ratio: float = 0.35):
    """
    在地图前方区域（上方 front_ratio 部分）滑动 block×block 窗口，
    选择障碍最少（空白最多）的块，取块中心作为终点。并列时取更靠前、更居中的块。
    """
    H, W = grid.shape
    block = max(1, min(block, H, W))
    rows = max(block, int(round(H * front_ratio)))
    sub = grid[:rows].astype(np.int32)
    ii = np.pad(sub.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[block:, block:] - ii[:-block, block:] - ii[block:, :-block] + ii[:-block, :-block]
    rr, cc = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    center_c = cc + block / 2
    score = s * 1000.0 + rr * 1.0 + np.abs(center_c - W / 2) * 0.5
    r, c = np.unravel_index(np.argmin(score), score.shape)
    return nearest_free(grid, (int(r + block // 2), int(c + block // 2)))


def select_goal(gm: GridMap, persons_xy=(), block: int = 8):
    """空中发现人员 -> 以置信度最高的人员为终点；否则选前方最空旷的块"""
    if persons_xy:
        x, y, _ = max(persons_xy, key=lambda p: p[2])
        cell = gm.world_to_cell(x, y)
        return nearest_free(gm.grid, cell), "person"
    return select_goal_free_block(gm.grid, block=block), "free_block"
