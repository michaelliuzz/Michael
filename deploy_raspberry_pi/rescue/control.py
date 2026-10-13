"""
小车沿期望航向行驶的控制代码

    PurePursuit        A* 航点 -> 前视点 -> 全局目标方位
    HeadingController  期望航向 -> PID -> (线速度 v, 角速度 ω) -> 左右履带/轮速度
    MotorDriver        电机驱动抽象：GpioZeroMotorDriver（树莓派 GPIO + 电机驱动板）/ DummyMotorDriver（仿真）

位姿约定: (x, y, theta)，theta 为车头相对地面坐标系 +x 轴的逆时针角（弧度）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PID:
    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 out_limit: float = math.inf, i_limit: float = math.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit = out_limit, i_limit
        self.reset()

    def reset(self):
        self._i = 0.0
        self._prev = None

    def update(self, err: float, dt: float) -> float:
        dt = max(dt, 1e-3)
        self._i = float(np.clip(self._i + err * dt, -self.i_limit, self.i_limit))
        d = 0.0 if self._prev is None else (err - self._prev) / dt
        self._prev = err
        out = self.kp * err + self.ki * self._i + self.kd * d
        return float(np.clip(out, -self.out_limit, self.out_limit))


@dataclass
class DriveConfig:
    track_width: float = 0.20      # 左右履带/轮中心距（米）
    max_speed: float = 0.35        # 最大线速度（米/秒）
    max_omega: float = 1.5         # 最大角速度（弧度/秒）
    rotate_in_place_deg: float = 60.0  # 航向误差超过该值时原地转向
    slow_dist: float = 1.0         # 前方距离小于该值时开始减速


class HeadingController:
    def __init__(self, cfg: DriveConfig | None = None, pid: PID | None = None):
        self.cfg = cfg or DriveConfig()
        self.pid = pid or PID(kp=2.0, ki=0.05, kd=0.15, out_limit=self.cfg.max_omega, i_limit=1.0)

    def compute(self, heading_err_rad: float, dt: float, front_dist: float = math.inf):
        """返回 (v, ω)"""
        c = self.cfg
        omega = self.pid.update(heading_err_rad, dt)
        if abs(heading_err_rad) > math.radians(c.rotate_in_place_deg):
            return 0.0, omega
        v = c.max_speed * max(0.0, math.cos(heading_err_rad))
        if front_dist < c.slow_dist:
            v *= max(0.2, front_dist / c.slow_dist)
        return v, omega

    def to_wheels(self, v: float, omega: float):
        """差速运动学，返回归一化到 [-1, 1] 的左右轮速度（可直接作为 PWM 占空比）"""
        c = self.cfg
        vl = v - omega * c.track_width / 2
        vr = v + omega * c.track_width / 2
        scale = max(1.0, abs(vl) / c.max_speed, abs(vr) / c.max_speed)
        return vl / c.max_speed / scale, vr / c.max_speed / scale


class PurePursuit:
    """沿 A* 航点前进：返回前视点在车体坐标系下的方位角"""

    def __init__(self, waypoints_xy, lookahead: float = 0.8, goal_tol: float = 0.3):
        self.set_path(waypoints_xy)
        self.lookahead = lookahead
        self.goal_tol = goal_tol

    def set_path(self, waypoints_xy):
        pts = np.asarray(waypoints_xy, float)
        # 航点之间加密，便于连续地找前视点
        dense = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.linalg.norm(b - a) / 0.05))
            dense.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        self.path = np.asarray(dense)
        self.idx = 0

    @property
    def goal(self):
        return self.path[-1]

    def reached(self, pose) -> bool:
        return math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1]) < self.goal_tol

    def target_bearing(self, pose) -> float:
        x, y, th = pose
        d = np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)
        # 只向前搜索，避免在回环路径上跳回
        win = slice(self.idx, min(len(d), self.idx + 200))
        self.idx = self.idx + int(np.argmin(d[win]))
        ahead = np.nonzero(d[self.idx:] >= self.lookahead)[0]
        tgt = self.path[self.idx + ahead[0]] if len(ahead) else self.path[-1]
        return wrap_angle(math.atan2(tgt[1] - y, tgt[0] - x) - th)


def recovery(rel_deg, ranges, target, ctrl, dt, cfg):
    """前方无可通行缝隙时的脱困策略：前方贴近障碍则缓慢后退，否则原地转向目标"""
    rel = np.asarray(rel_deg, float)
    rr = np.asarray(ranges, float)
    ok = ~np.isnan(rr) & (rr > 0.02)   # 丢弃无效测量（RPLIDAR 用 0 表示测量失败）
    rel, rr = rel[ok], rr[ok]
    front = np.abs(rel) <= 90
    near = cfg.car_width / 2 + cfg.margin + 0.1
    back_clear = not np.any((np.abs(rel) >= 135) & (rr < 0.35))
    if np.any(front & (rr < near)) and back_clear:
        return -0.08, 0.0
    err = target if abs(target) > 0.3 else math.copysign(math.pi / 2, target or 1.0)
    return 0.0, ctrl.compute(err, dt)[1]


class MotorDriver:
    def set(self, left: float, right: float):
        raise NotImplementedError

    def stop(self):
        self.set(0.0, 0.0)

    def close(self):
        self.stop()


class DummyMotorDriver(MotorDriver):
    """仿真 / 无硬件时使用：只记录指令"""

    def __init__(self):
        self.last = (0.0, 0.0)

    def set(self, left, right):
        self.last = (float(left), float(right))


class GpioZeroMotorDriver(MotorDriver):
    """
    树莓派 GPIO + 双路电机驱动板（如 L298N / TB6612）
    引脚为 BCM 编号，需按实际接线修改；negative 值表示反转
    """

    def __init__(self, left_pins=(17, 27), right_pins=(22, 23), left_pwm=None, right_pwm=None):
        from gpiozero import Motor  # 仅在真机上导入
        self.left = Motor(*left_pins, enable=left_pwm)
        self.right = Motor(*right_pins, enable=right_pwm)

    @staticmethod
    def _apply(m, v):
        v = float(np.clip(v, -1, 1))
        if v > 0:
            m.forward(v)
        elif v < 0:
            m.backward(-v)
        else:
            m.stop()

    def set(self, left, right):
        self._apply(self.left, left)
        self._apply(self.right, right)

    def close(self):
        self.stop()
        self.left.close()
        self.right.close()
