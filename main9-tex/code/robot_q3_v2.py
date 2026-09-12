from __future__ import annotations
import argparse
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
ARENA_RADIUS = 1800.0
BEARING_ERROR_DEG = 1.0
CLEAR_RADIUS = 20.0
CHANNELS = tuple(range(1, 21))

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

@dataclass(frozen=True)
class Bearing:
    station: Point
    angle_deg: float

@dataclass
class ChannelState:
    bearings: list[Bearing] = field(default_factory=list)
    cleared: bool = False
    polygon: list[Point] = field(default_factory=list)

class RobotAPI(Protocol):

    def enter(self) -> dict:
        ...

    def measure(self, point: Point, channel: int) -> dict:
        ...

    def clear(self, point: Point, channel: int) -> dict:
        ...

    def exit(self) -> dict:
        ...

class ActionRejected(RuntimeError):
    pass

def require_accepted(response: dict, action: str) -> None:
    if response.get('accepted') is not True:
        raise ActionRejected(f'{action} 未被接受：{response}')

class HttpRobotAPI:

    def __init__(self, base_url: str, robot_id: str, log_path: Path):
        self.base_url = base_url.rstrip('/')
        self.robot_id = robot_id
        self.log_path = log_path
        self.counter = 0

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f'{prefix}-{self.counter}-{uuid.uuid4().hex[:8]}'

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = Request(self.base_url + path, data=body, headers={'Content-Type': 'application/json'}, method='POST')
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(req, timeout=8) as response:
                    result = json.loads(response.read().decode('utf-8'))
                self._log(path, payload, result)
                return result
            except (URLError, TimeoutError, HTTPError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * 2 ** attempt)
        raise RuntimeError(f' 请求 {path} 失败：{last_error}')

    def _base(self, request_id: str) -> dict:
        return {'arena_id': 'default', 'robot_id': self.robot_id, 'request_id': request_id}

    def _log(self, path: str, request: dict, response: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {'path': path, 'request': request, 'response': response}
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def enter(self) -> dict:
        return self._post('/enter', self._base(self._id('enter')))

    def measure(self, point: Point, channel: int) -> dict:
        p = self._base(self._id('measure'))
        p.update({'position': {'x': point.x, 'y': point.y}, 'channel': channel})
        return self._post('/measure', p)

    def clear(self, point: Point, channel: int) -> dict:
        p = self._base(self._id('clear'))
        p.update({'position': {'x': point.x, 'y': point.y}, 'channel': channel})
        return self._post('/clear', p)

    def exit(self) -> dict:
        return self._post('/exit', self._base(self._id('exit')))

def regular_polygon(radius: float, sides: int=720) -> list[Point]:
    return [Point(radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides)) for i in range(sides)]

def cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx

def clip_halfplane(poly: list[Point], station: Point, angle_deg: float, keep_left: bool) -> list[Point]:
    if not poly:
        return []
    a = math.radians(angle_deg)
    dx, dy = (math.cos(a), math.sin(a))

    def signed(p: Point) -> float:
        return cross(dx, dy, p.x - station.x, p.y - station.y)

    def inside(p: Point) -> bool:
        value = signed(p)
        return value >= -1e-08 if keep_left else value <= 1e-08
    result: list[Point] = []
    previous = poly[-1]
    previous_in = inside(previous)
    for current in poly:
        current_in = inside(current)
        if current_in != previous_in:
            sx, sy = (current.x - previous.x, current.y - previous.y)
            denominator = cross(dx, dy, sx, sy)
            if abs(denominator) > 1e-12:
                t = -signed(previous) / denominator
                result.append(Point(previous.x + t * sx, previous.y + t * sy))
        if current_in:
            result.append(current)
        previous, previous_in = (current, current_in)
    return result

def polygon_from_bearings(bearings: Iterable[Bearing]) -> list[Point]:
    poly = regular_polygon(ARENA_RADIUS)
    for bearing in bearings:
        poly = clip_halfplane(poly, bearing.station, bearing.angle_deg - BEARING_ERROR_DEG, keep_left=True)
        poly = clip_halfplane(poly, bearing.station, bearing.angle_deg + BEARING_ERROR_DEG, keep_left=False)
    return poly

def min_dist_to_polygon(p: Point, poly: list[Point]) -> float:
    best = float('inf')
    n = len(poly)
    for i in range(n):
        a, b = (poly[i], poly[(i + 1) % n])
        abx, aby = (b.x - a.x, b.y - a.y)
        apx, apy = (p.x - a.x, p.y - a.y)
        denom = abx * abx + aby * aby
        t = 0.0 if denom <= 1e-12 else max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        d = math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))
        if d < best:
            best = d
    return best

def circle_from_two(a: Point, b: Point) -> tuple[Point, float]:
    center = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
    return (center, center.distance(a))

def circle_from_three(a: Point, b: Point, c: Point) -> tuple[Point, float] | None:
    d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if abs(d) < 1e-09:
        return None
    aa, bb, cc = (a.x * a.x + a.y * a.y, b.x * b.x + b.y * b.y, c.x * c.x + c.y * c.y)
    ux = (aa * (b.y - c.y) + bb * (c.y - a.y) + cc * (a.y - b.y)) / d
    uy = (aa * (c.x - b.x) + bb * (a.x - c.x) + cc * (b.x - a.x)) / d
    center = Point(ux, uy)
    return (center, center.distance(a))

def covers(center: Point, radius: float, points: Iterable[Point]) -> bool:
    return all((center.distance(p) <= radius + 1e-06 for p in points))

def minimum_enclosing_circle(points: list[Point]) -> tuple[Point, float]:
    if not points:
        raise ValueError(' 空定位区域')
    if len(points) == 1:
        return (points[0], 0.0)
    best: tuple[Point, float] | None = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            candidate = circle_from_two(points[i], points[j])
            if covers(*candidate, points) and (best is None or candidate[1] < best[1]):
                best = candidate
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                candidate = circle_from_three(points[i], points[j], points[k])
                if candidate and covers(*candidate, points) and (best is None or candidate[1] < best[1]):
                    best = candidate
    if best is None:
        raise RuntimeError(' 未能构造最小覆盖圆')
    return best

def discovery_points() -> list[Point]:
    points = [Point(0.0, 0.0)]
    for i in range(8):
        angle = 2 * math.pi * i / 8
        points.append(Point(1000 * math.cos(angle), 1000 * math.sin(angle)))
    return points

def serpentine_channels(index: int) -> tuple[int, ...]:
    return CHANNELS if index % 2 == 0 else tuple(reversed(CHANNELS))

def choose_refinement_point(center: Point, state: ChannelState, fallback_center: Point | None=None, az_offset_deg: float=0.0) -> Point:
    if fallback_center is not None:
        last = state.bearings[-1]
        return Point((fallback_center.x + center.x) / 2, (fallback_center.y + center.y) / 2)
    last = state.bearings[-1]
    angle = math.radians(last.angle_deg + 90.0 + az_offset_deg)
    step = 450.0
    candidates = [Point(center.x + step * math.cos(angle), center.y + step * math.sin(angle)), Point(center.x - step * math.cos(angle), center.y - step * math.sin(angle))]
    valid = [p for p in candidates if math.hypot(p.x, p.y) <= ARENA_RADIUS]
    if not valid:
        return center
    return min(valid, key=lambda p: p.distance(last.station))

def optimized_clear_order(start: Point, targets: dict[int, Point]) -> list[int]:
    remaining = set(targets)
    route: list[int] = []
    current = start
    while remaining:
        channel = min(remaining, key=lambda c: current.distance(targets[c]))
        route.append(channel)
        remaining.remove(channel)
        current = targets[channel]

    def length(order: list[int]) -> float:
        if not order:
            return 0.0
        total = start.distance(targets[order[0]])
        total += sum((targets[a].distance(targets[b]) for a, b in zip(order, order[1:])))
        return total
    improved = True
    best = length(route)
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                value = length(candidate)
                if value + 1e-06 < best:
                    route, best, improved = (candidate, value, True)
    return route

class Q3Strategy:
    CLEAR_RESERVE_S = 150.0
    REFINE_RESERVE_S = 300.0
    INSITE_CLEAR_M = 400.0

    def __init__(self, api: RobotAPI):
        self.api = api
        self.states = {channel: ChannelState() for channel in CHANNELS}
        self.deadline: float | None = None
        self.t0: float = 0.0

    def time_left(self) -> float:
        if self.deadline is None:
            return float('inf')
        return self.deadline - time.monotonic()

    def run(self) -> dict:
        self.t0 = time.monotonic()
        entered = self.api.enter()
        if entered.get('accepted') is not True:
            raise RuntimeError(f'/enter 未被接受：{entered}')
        remaining = entered.get('remaining_real_duration_s')
        if isinstance(remaining, (int, float)) and remaining > 0:
            self.deadline = self.t0 + float(remaining)
        errors: list[str] = []
        search_points = discovery_points()
        current_point = Point(0.0, 0.0)
        try:
            for point_index, point in enumerate(search_points):
                for channel in serpentine_channels(point_index):
                    state = self.states[channel]
                    if state.cleared:
                        continue
                    if state.bearings:
                        state.polygon = polygon_from_bearings(state.bearings)
                        if state.polygon and min_dist_to_polygon(point, state.polygon) > 1500.0:
                            continue
                    response = self.api.measure(point, channel)
                    current_point = point
                    require_accepted(response, f' 测量 (点 {point_index}, 频道 {channel})')
                    result = response.get('measure_result')
                    if result == 'direction':
                        self.states[channel].bearings.append(Bearing(point, float(response['svd_deg'])))
                    elif result == 'near':
                        clear_response = self.api.clear(point, channel)
                        require_accepted(clear_response, f' 清除 (点 {point_index}, 频道 {channel})')
                        self.states[channel].cleared = clear_response.get('clear_result') == 'success'
                if self.time_left() < self.REFINE_RESERVE_S + self.CLEAR_RESERVE_S:
                    break
        except (ActionRejected, RuntimeError) as exc:
            errors.append(f' 覆盖搜索中断：{exc}')
        ready: dict[int, Point] = {}
        failed: list[int] = []
        pending = {c for c, s in self.states.items() if s.bearings and (not s.cleared)}
        attempts = {c: 0 for c in pending}
        fallback_from: dict[int, Point] = {}
        try:
            while pending and self.time_left() >= self.REFINE_RESERVE_S:
                candidates: dict[int, tuple[Point, Point, float]] = {}
                finished: list[int] = []
                for channel in pending:
                    state = self.states[channel]
                    state.polygon = polygon_from_bearings(state.bearings)
                    if not state.polygon:
                        failed.append(channel)
                        finished.append(channel)
                        continue
                    center, radius = minimum_enclosing_circle(state.polygon)
                    if radius <= CLEAR_RADIUS - 0.25:
                        if current_point.distance(center) <= self.INSITE_CLEAR_M:
                            try:
                                clr = self.api.clear(center, channel)
                                require_accepted(clr, f' 就地清除 (频道 {channel})')
                                if clr.get('clear_result') == 'success':
                                    state.cleared = True
                                else:
                                    ready[channel] = center
                            except (ActionRejected, RuntimeError) as exc:
                                errors.append(f' 频道 {channel} 就地清除失败：{exc}')
                                ready[channel] = center
                        else:
                            ready[channel] = center
                        finished.append(channel)
                        continue
                    if attempts[channel] >= 8:
                        failed.append(channel)
                        finished.append(channel)
                        continue
                    if channel in fallback_from:
                        point = choose_refinement_point(center, state, fallback_center=fallback_from[channel])
                    else:
                        point = Point(center.x, center.y)
                    candidates[channel] = (point, center, radius)
                for channel in finished:
                    pending.discard(channel)
                if not candidates:
                    break
                channel = min(candidates, key=lambda c: current_point.distance(candidates[c][0]))
                next_point = candidates[channel][0]
                state = self.states[channel]
                try:
                    response = self.api.measure(next_point, channel)
                    require_accepted(response, f' 补测 (频道 {channel})')
                except (ActionRejected, RuntimeError) as exc:
                    errors.append(f' 频道 {channel} 补测失败：{exc}')
                    failed.append(channel)
                    pending.discard(channel)
                    continue
                current_point = next_point
                attempts[channel] += 1
                result = response.get('measure_result')
                if result == 'direction':
                    state.bearings.append(Bearing(next_point, float(response['svd_deg'])))
                    fallback_from.pop(channel, None)
                elif result == 'near':
                    try:
                        clear_response = self.api.clear(next_point, channel)
                        require_accepted(clear_response, f' 清除 (频道 {channel})')
                    except (ActionRejected, RuntimeError) as exc:
                        errors.append(f' 频道 {channel} 清除失败：{exc}')
                        failed.append(channel)
                        pending.discard(channel)
                        continue
                    state.cleared = clear_response.get('clear_result') == 'success'
                    if not state.cleared:
                        failed.append(channel)
                    pending.discard(channel)
                elif result == 'no_signal':
                    fallback_from[channel] = next_point
                else:
                    errors.append(f' 频道 {channel} 未知测量结果：{response}')
                    failed.append(channel)
                    pending.discard(channel)
        except (ActionRejected, RuntimeError) as exc:
            errors.append(f' 定位阶段中断：{exc}')
        for channel in pending:
            if not self.states[channel].cleared and channel not in ready:
                failed.append(channel)
        clear_failed: list[int] = []
        clear_order = optimized_clear_order(current_point, ready)
        for channel in clear_order:
            if self.time_left() <= self.CLEAR_RESERVE_S * 0.3:
                break
            clear_point = ready.pop(channel)
            try:
                response = self.api.clear(clear_point, channel)
                require_accepted(response, f' 清除 (频道 {channel})')
            except (ActionRejected, RuntimeError) as exc:
                errors.append(f' 频道 {channel} 清除失败：{exc}')
                clear_failed.append(channel)
                continue
            current_point = clear_point
            self.states[channel].cleared = response.get('clear_result') == 'success'
            if not self.states[channel].cleared:
                clear_failed.append(channel)
        clear_failed.extend(ready.keys())
        summary = {'discovered_channels': [c for c, s in self.states.items() if s.bearings or s.cleared], 'cleared_channels': [c for c, s in self.states.items() if s.cleared], 'failed_channels': sorted((c for c in set(failed + clear_failed) if not self.states[c].cleared)), 'unfound_channels': sorted((c for c, s in self.states.items() if not s.bearings and (not s.cleared))), 'real_time_used_s': round(time.monotonic() - self.t0, 1), 'time_left_s': round(max(self.time_left(), 0.0), 1)}
        if errors:
            summary['action_errors'] = errors
        try:
            exit_response = self.api.exit()
            require_accepted(exit_response, '/exit')
            summary['exit'] = exit_response
            summary['exit_accepted'] = True
        except (ActionRejected, RuntimeError) as exc:
            summary['exit'] = {'error': str(exc)}
            summary['exit_accepted'] = False
        return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help=' 连接官方模拟器；默认不连接')
    parser.add_argument('--robot-id', default='', help=' 当前登录的参赛队号')
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--log', default='robot-actions.jsonl')
    args = parser.parse_args()
    if not args.live:
        print('dry-run：未连接模拟器。使用 --live --robot-id < 参赛队号> 才会执行测试。')
        return
    if not args.robot_id:
        raise SystemExit('live 模式必须提供 --robot-id')
    api = HttpRobotAPI(args.base_url, args.robot_id, Path(args.log))
    print(json.dumps(Q3Strategy(api).run(), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
