"""
研究内容(1)：小车在前方有障碍物时判断可通行路段 —— 基于二维激光雷达的缝隙检测

算法（与申报书描述一致）:
    1. 对一圈扫描数据按角度排序、分箱，得到前方 120° 范围内随角度变化的距离曲线 d(θ)
    2. 设定安全距离阈值（默认 1 m），d(θ) < 阈值的点视为“近距离障碍物边界”
    3. 相邻的非障碍角度段构成“缝隙”，每个缝隙用 起始角 / 结束角 / 平均距离 描述
    4. 由缝隙左右边界障碍点计算缝隙实际宽度，宽度 >= 车宽 + 安全冗余 的缝隙判为可通行
    5. 在可通行缝隙中，选择最接近全局目标方位（来自 A* 路径）的方向作为期望航向，
       并让航向与缝隙边界保持车体半宽对应的角度余量
    6. 走廊校验：沿期望航向车体扫过的矩形内不得有任何扫描点（含侧方点）

    前方无可行航向时由 control.recovery 执行脱困（贴近障碍则后退，否则原地转向）

角度约定: 0° = 车头正前方，逆时针（向左）为正，单位为度；距离单位为米
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GapConfig:
    safe_dist: float = 1.0       # 安全距离阈值（米）
    fov_deg: float = 120.0       # 分析的前方扇区
    car_width: float = 0.25      # 小车宽度（米）
    margin: float = 0.10         # 单侧安全冗余（米）
    bin_deg: float = 1.0         # 角度分箱宽度
    max_range: float = 8.0       # 雷达最大量程，无回波按此处理
    stop_dist: float = 0.25      # 正前方急停距离
    stop_cone_deg: float = 30.0  # 急停检测扇区（全角）

    @property
    def required_width(self) -> float:
        return self.car_width + 2 * self.margin


@dataclass
class Gap:
    start_deg: float           # 右边界角（较小角度）
    end_deg: float             # 左边界角（较大角度）
    mean_dist: float           # 缝隙内平均距离
    width_m: float             # 缝隙实际宽度
    passable: bool
    bounded_right: bool = True  # 右侧是否有真实障碍（False 表示延伸到扇区边缘）
    bounded_left: bool = True
    right_dist: float = math.inf  # 右/左边界障碍点距离
    left_dist: float = math.inf

    @property
    def center_deg(self) -> float:
        return (self.start_deg + self.end_deg) / 2


@dataclass
class LocalPlan:
    heading_deg: float | None    # 期望航向；None 表示前方无可通行缝隙
    gaps: list[Gap] = field(default_factory=list)
    emergency_stop: bool = False
    min_front_dist: float = math.inf
    profile: tuple[np.ndarray, np.ndarray] | None = None


def rplidar_to_standard(angle_deg: np.ndarray) -> np.ndarray:
    """RPLIDAR 角度为顺时针 0~360°，转换为本模块约定（逆时针为正，范围 -180~180）"""
    a = -np.asarray(angle_deg, float)
    return (a + 180.0) % 360.0 - 180.0


def scan_to_profile(angles_deg, ranges, cfg: GapConfig):
    """
    把原始扫描点整理成前方扇区内的距离曲线 d(θ)
    inf 表示该方向无回波（开阔），按最大量程处理；
    0 / NaN 为无效测量（RPLIDAR 用 0 表示测量失败）被丢弃，没有任何有效测量的角度箱用相邻箱插值
    """
    a = np.asarray(angles_deg, float)
    d = np.asarray(ranges, float)
    ok = np.isfinite(a) & ~np.isnan(d) & (d > 0.02)
    a, d = a[ok], np.minimum(d[ok], cfg.max_range)
    half = cfg.fov_deg / 2
    nb = int(round(cfg.fov_deg / cfg.bin_deg))
    centers = -half + (np.arange(nb) + 0.5) * cfg.bin_deg
    prof = np.full(nb, np.nan)
    sel = (a >= -half) & (a < half)
    if sel.any():
        idx = np.clip(((a[sel] + half) / cfg.bin_deg).astype(int), 0, nb - 1)
        prof_min = np.full(nb, np.inf)
        np.minimum.at(prof_min, idx, d[sel])
        prof = np.where(np.isinf(prof_min), np.nan, prof_min)
    valid = ~np.isnan(prof)
    if not valid.any():
        prof[:] = cfg.max_range
    elif not valid.all():
        prof[~valid] = np.interp(centers[~valid], centers[valid], prof[valid])
    return centers, prof


def _polar_dist(r1, a1_deg, r2, a2_deg) -> float:
    da = math.radians(a2_deg - a1_deg)
    return math.sqrt(max(0.0, r1 * r1 + r2 * r2 - 2 * r1 * r2 * math.cos(da)))


def find_gaps(centers: np.ndarray, prof: np.ndarray, cfg: GapConfig) -> list[Gap]:
    near = prof < cfg.safe_dist
    gaps = []
    n = len(prof)
    i = 0
    half_bin = cfg.bin_deg / 2
    while i < n:
        if near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not near[j + 1]:
            j += 1
        start, end = centers[i] - half_bin, centers[j] + half_bin
        # 边界点：缝隙两侧紧邻的近距离障碍点；延伸到扇区边缘时按安全距离处保守估计
        bounded_r, bounded_l = i > 0, j < n - 1
        r_r, a_r = (prof[i - 1], centers[i - 1]) if bounded_r else (cfg.safe_dist, start)
        r_l, a_l = (prof[j + 1], centers[j + 1]) if bounded_l else (cfg.safe_dist, end)
        width = _polar_dist(r_r, a_r, r_l, a_l)
        gaps.append(Gap(float(start), float(end), float(np.mean(prof[i:j + 1])), width,
                        width >= cfg.required_width, bounded_r, bounded_l, float(r_r), float(r_l)))
        i = j + 1
    return gaps


def corridor_clear(heading_deg: float, pts_deg: np.ndarray, pts_dist: np.ndarray,
                   cfg: GapConfig, length: float | None = None) -> bool:
    """沿 heading 前进 length 米时，车体扫过的矩形（半宽 = 车半宽 + 冗余）内是否没有任何扫描点"""
    length = cfg.safe_dist if length is None else length
    half = cfg.car_width / 2 + cfg.margin
    a = np.radians(pts_deg - heading_deg)
    fx = pts_dist * np.cos(a)
    fy = pts_dist * np.sin(a)
    return not np.any((fx > 0) & (fx < length) & (np.abs(fy) < half))


def choose_heading(gaps: list[Gap], target_deg: float, cfg: GapConfig,
                   pts_deg=None, pts_dist=None) -> float | None:
    """
    在可通行缝隙中选取最接近目标方位的期望航向。
    1) 航向与每侧边界障碍保持角度余量 asin((车半宽+冗余)/边界距离)，余量占满缝隙时该缝隙不可用；
    2) 若给出全部扫描点，再检查车体沿该航向扫过的矩形走廊内无障碍点（防止侧前方近处障碍被刮碰）。
    """
    half = cfg.car_width / 2 + cfg.margin

    def clearance(dist):
        return 90.0 if dist <= half else math.degrees(math.asin(half / dist))

    candidates = []
    for g in gaps:
        if not g.passable:
            continue
        lo = g.start_deg + (clearance(g.right_dist) if g.bounded_right else 0.0)
        hi = g.end_deg - (clearance(g.left_dist) if g.bounded_left else 0.0)
        if lo > hi:
            continue
        h0 = min(max(target_deg, lo), hi)
        candidates.append(h0)
        candidates.extend(np.arange(lo, hi + 1e-9, 2.0).tolist())
    candidates.sort(key=lambda h: abs(h - target_deg))
    for h in candidates:
        if pts_deg is None or corridor_clear(h, pts_deg, pts_dist, cfg):
            return float(h)
    return None


def plan_local(angles_deg, ranges, target_deg: float, cfg: GapConfig | None = None) -> LocalPlan:
    """
    一次完整的局部决策：扫描 -> d(θ) -> 缝隙 -> 期望航向

    Args:
        target_deg: 全局目标方位（由 A* 路径前视点给出，车体坐标系）
    """
    cfg = cfg or GapConfig()
    centers, prof = scan_to_profile(angles_deg, ranges, cfg)
    gaps = find_gaps(centers, prof, cfg)
    cone = np.abs(centers) <= cfg.stop_cone_deg / 2
    min_front = float(prof[cone].min()) if cone.any() else math.inf
    a = np.asarray(angles_deg, float)
    d = np.asarray(ranges, float)
    ok = np.isfinite(a) & np.isfinite(d) & (d > 0.02)
    heading = choose_heading(gaps, target_deg, cfg, a[ok], d[ok])
    return LocalPlan(heading, gaps, min_front < cfg.stop_dist, min_front, (centers, prof))
