"""
研究内容(4)：利用摄像头内参与小车上的安装高度，结合检测框像素位置，估算人体距离与相对方位

方法一（地面接触点法，主方法）:
    检测框底边中点视为人体与地面的接触点，将该像素反投影为射线，
    与地平面求交，得到人体在小车坐标系下的位置 (X 前, Y 左)。站立、倒地人体均适用。
方法二（身高先验法，备用）:
    框底边被图像下沿截断或射线不与地面相交时，按 距离 ≈ fy · H_person / 框高 估计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CarCamera:
    width: int = 640
    height: int = 480
    hfov_deg: float = 62.2
    mount_height_m: float = 0.15   # 摄像头光心离地高度
    pitch_down_deg: float = 10.0   # 摄像头向下俯角
    yaw_offset_deg: float = 0.0    # 摄像头相对车头的安装偏角（左为正）

    @property
    def f(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)


@dataclass
class PersonFix:
    x: float          # 车体坐标系：前方距离（米）
    y: float          # 车体坐标系：左侧偏移（米）
    distance: float
    bearing_deg: float  # 左为正
    method: str


def locate_person(xyxy, cam: CarCamera, person_height_m: float = 1.7,
                  bottom_margin_px: float = 3.0) -> PersonFix | None:
    x1, y1, x2, y2 = xyxy
    f = cam.f
    u = (x1 + x2) / 2
    xn = (u - cam.width / 2) / f
    yaw_off = math.radians(cam.yaw_offset_deg)
    th = math.radians(cam.pitch_down_deg)

    truncated = y2 >= cam.height - bottom_margin_px
    if not truncated:
        yn = (y2 - cam.height / 2) / f
        denom = math.sin(th) + yn * math.cos(th)
        if denom > 1e-6:
            t = cam.mount_height_m / denom
            X = t * (math.cos(th) - yn * math.sin(th))
            Y = -t * xn
            return _fix(X, Y, yaw_off, "ground_contact")

    h_px = y2 - y1
    if h_px <= 1:
        return None
    dist = f * person_height_m / h_px
    bearing = math.atan2(-xn, 1.0)
    return _fix(dist * math.cos(bearing), dist * math.sin(bearing), yaw_off, "height_prior")


def _fix(X, Y, yaw_off, method) -> PersonFix:
    c, s = math.cos(yaw_off), math.sin(yaw_off)
    X, Y = c * X - s * Y, s * X + c * Y
    return PersonFix(X, Y, math.hypot(X, Y), math.degrees(math.atan2(Y, X)), method)


def car_to_world(fix: PersonFix, pose):
    """车体坐标 -> 地面坐标，pose = (x, y, theta)"""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return x + c * fix.x - s * fix.y, y + s * fix.x + c * fix.y
