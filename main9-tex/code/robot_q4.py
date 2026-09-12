from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from robot_q3_v2 import ARENA_RADIUS, BEARING_ERROR_DEG, CHANNELS, CLEAR_RADIUS, ActionRejected, Point, Bearing, ChannelState, HttpRobotAPI, clip_halfplane, polygon_from_bearings, min_dist_to_polygon, minimum_enclosing_circle, regular_polygon, require_accepted, optimized_clear_order

def ring_points(n: int, r: float, phase_deg: float=0.0) -> list[Point]:
    return [Point(r * math.cos(2 * math.pi * i / n + math.radians(phase_deg)), r * math.sin(2 * math.pi * i / n + math.radians(phase_deg))) for i in range(n)]

def search_points() -> list[Point]:
    pts = [Point(0.0, 0.0)]
    for k in range(13):
        a = 2.0 * math.pi * k / 13.0
        pts.append(Point(1000.0 * math.cos(a), 1000.0 * math.sin(a)))
    for k in range(12):
        a = 2.0 * math.pi * k / 12.0
        pts.append(Point(1900.0 * math.cos(a), 1900.0 * math.sin(a)))
    return pts

def search_points_tri27() -> list[Point]:
    pts = [(-1500.0, -1800.0), (-500.0, -1800.0), (500.0, -1800.0), (1500.0, -1800.0), (-2000.0, -933.9746), (-1000.0, -933.9746), (0.0, -933.9746), (1000.0, -933.9746), (2000.0, -933.9746), (-2500.0, -67.9492), (-1500.0, -67.9492), (-500.0, -67.9492), (500.0, -67.9492), (1500.0, -67.9492), (2500.0, -67.9492), (-2000.0, 798.0762), (-1000.0, 798.0762), (0.0, 798.0762), (1000.0, 798.0762), (2000.0, 798.0762), (-1500.0, 1664.1016), (-500.0, 1664.1016), (500.0, 1664.1016), (1500.0, 1664.1016), (-1000.0, 2530.127), (0.0, 2530.127), (1000.0, 2530.127)]
    order = [(-500.0, -67.9492), (-1500.0, -67.9492), (-2500.0, -67.9492), (-2000.0, 798.0762), (-1000.0, 798.0762), (-1500.0, 1664.1016), (-1000.0, 2530.127), (0.0, 2530.127), (1000.0, 2530.127), (1500.0, 1664.1016), (500.0, 1664.1016), (-500.0, 1664.1016), (0.0, 798.0762), (500.0, -67.9492), (1000.0, 798.0762), (1500.0, -67.9492), (2000.0, 798.0762), (2500.0, -67.9492), (2000.0, -933.9746), (1000.0, -933.9746), (0.0, -933.9746), (-1000.0, -933.9746), (-2000.0, -933.9746), (-1500.0, -1800.0), (-500.0, -1800.0), (500.0, -1800.0), (1500.0, -1800.0)]
    pts_set = {tuple(p) for p in pts}
    order_set = {tuple(p) for p in order}
    assert pts_set == order_set and len(order) == len(pts), ' 巡视路径与测点集不一致'
    return [Point(x, y) for x, y in order]

def serpentine_channels(index: int) -> tuple[int, ...]:
    return CHANNELS if index % 2 == 0 else tuple(reversed(CHANNELS))
DETOUR_MAX = 400.0
FINE_GRID_SPACING = 17.2
FINE_GRID_PAD = 20.0
FINE_GRID_RADIUS = 1820.0
FINE_GRID_RANGE_CLIP = 1520.0
FINE_GRID_RESERVE_S = 180.0

def _point_in_convex_poly(p: Point, poly: list[Point]) -> bool:
    pos = neg = False
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        if cross > 1e-09:
            pos = True
        elif cross < -1e-09:
            neg = True
    return not (pos and neg)

def fine_grid_points(bearings: list[Bearing]) -> list[Point]:
    if not bearings:
        return []
    station = bearings[0].station
    poly = polygon_from_bearings(bearings)
    if not poly or len(poly) < 3:
        return []
    angle = math.radians(bearings[0].angle_deg)
    ex = (math.cos(angle), math.sin(angle))
    ey = (-math.sin(angle), math.cos(angle))
    sx, sy = (station.x, station.y)
    local = [((p.x - sx) * ex[0] + (p.y - sy) * ex[1], (p.x - sx) * ey[0] + (p.y - sy) * ey[1]) for p in poly]
    pad = FINE_GRID_PAD
    xmin = min((p[0] for p in local)) - pad
    xmax = max((p[0] for p in local)) + pad
    ymin = min((p[1] for p in local)) - pad
    ymax = max((p[1] for p in local)) + pad
    spacing = FINE_GRID_SPACING
    gap = spacing * math.sqrt(3.0) / 2.0
    raw: list[tuple[int, float, Point]] = []
    for row in range(math.floor(ymin / gap) - 1, math.ceil(ymax / gap) + 2):
        ly = row * gap
        for col in range(math.floor(xmin / spacing - 0.5 * row) - 1, math.ceil(xmax / spacing - 0.5 * row) + 2):
            lx = spacing * (col + 0.5 * row)
            px = sx + lx * ex[0] + ly * ey[0]
            py = sy + lx * ex[1] + ly * ey[1]
            if px * px + py * py > FINE_GRID_RADIUS * FINE_GRID_RADIUS + 1e-09:
                continue
            if math.hypot(px - sx, py - sy) > FINE_GRID_RANGE_CLIP:
                continue
            p = Point(px, py)
            if _point_in_convex_poly(p, poly) or min_dist_to_polygon(p, poly) <= pad + 1e-09:
                raw.append((row, lx, p))
    ordered: list[Point] = []
    for index, row in enumerate(sorted({item[0] for item in raw})):
        row_points = sorted((item for item in raw if item[0] == row), key=lambda item: item[1], reverse=bool(index % 2))
        ordered.extend((item[2] for item in row_points))
    return ordered

class Q4Strategy:
    CLEAR_RESERVE_S = 150.0
    REFINE_RESERVE_S = 300.0
    MAX_REFINE_ITERS = 16

    def __init__(self, api):
        self.api = api
        self.states = {channel: ChannelState() for channel in CHANNELS}
        self.search_ready: dict[int, Point] = {}
        self.deadline: float | None = None
        self.t0: float = 0.0
        self.deferred_errors: list[str] = []
        self.fine_grid_channels: list[int] = []

    def time_left(self) -> float:
        if self.deadline is None:
            return float('inf')
        return self.deadline - time.monotonic()

    def coverage_search(self) -> Point:
        current = Point(0.0, 0.0)
        self.current_pos = current
        for point_index, point in enumerate(search_points()):
            for channel in serpentine_channels(point_index):
                state = self.states[channel]
                if state.cleared or channel in self.search_ready:
                    continue
                discovered_count = sum((bool(s.bearings) or s.cleared for s in self.states.values()))
                if discovered_count >= 16 and (not state.bearings):
                    continue
                if state.bearings:
                    state.polygon = polygon_from_bearings(state.bearings)
                    if state.polygon and min_dist_to_polygon(point, state.polygon) > 1500.0:
                        continue
                response = self.api.measure(point, channel)
                current = point
                self.current_pos = current
                require_accepted(response, f' 测量 (点 {point_index}, 频道 {channel})')
                result = response.get('measure_result')
                if result == 'direction':
                    state.bearings.append(Bearing(point, float(response['svd_deg'])))
                    state.polygon = polygon_from_bearings(state.bearings)
                    if state.polygon:
                        center, radius = minimum_enclosing_circle(state.polygon)
                        if radius <= CLEAR_RADIUS - 0.25:
                            self.search_ready[channel] = center
                elif result == 'near':
                    clear_response = self.api.clear(point, channel)
                    require_accepted(clear_response, f' 清除 (点 {point_index}, 频道 {channel})')
                    state.cleared = clear_response.get('clear_result') == 'success'
            discovered_count = sum((bool(s.bearings) or s.cleared for s in self.states.values()))
            if discovered_count >= 16 and all((s.cleared or c in self.search_ready for c, s in self.states.items() if s.bearings)):
                break
            sp = search_points()
            if point_index + 1 < len(sp):
                nxt = sp[point_index + 1]
                for channel, center in list(self.search_ready.items()):
                    if self.states[channel].cleared:
                        continue
                    detour = point.distance(center) + center.distance(nxt) - point.distance(nxt)
                    if detour <= DETOUR_MAX:
                        try:
                            clr = self.api.clear(center, channel)
                            require_accepted(clr, f' 顺路清除 (频道 {channel})')
                            if clr.get('clear_result') == 'success':
                                self.states[channel].cleared = True
                                self.search_ready.pop(channel, None)
                        except (ActionRejected, RuntimeError) as exc:
                            self.deferred_errors.append(f' 顺路清除 (频道 {channel}) 失败：{exc}')
                            break
            if self.time_left() < self.REFINE_RESERVE_S + self.CLEAR_RESERVE_S:
                break
        return current

    def _ladder_candidates(self, state: ChannelState) -> list[Point]:
        b = state.bearings[0]
        a = math.radians(b.angle_deg)
        ux, uy = (math.cos(a), math.sin(a))
        nx, ny = (-uy, ux)
        pts = []
        for t in (0.0, 350.0, 700.0, 1050.0):
            for s in (450.0, -450.0):
                pts.append(Point(b.station.x + t * ux + s * nx, b.station.y + t * uy + s * ny))
        return pts

    def refine(self, channel: int) -> tuple[bool, Point | None]:
        state = self.states[channel]
        if not state.bearings:
            return (False, None)
        state.polygon = polygon_from_bearings(state.bearings)
        if not state.polygon:
            return (False, None)
        center, radius = minimum_enclosing_circle(state.polygon)
        if radius <= CLEAR_RADIUS - 0.25:
            return (True, center)
        no_signal_streak = 0
        az_offset = 0.0
        candidate: Point | None = None
        last_ok: Point | None = state.bearings[-1].station
        visited: list[tuple[float, float]] = []
        ladder = self._ladder_candidates(state)
        li = 0
        walk_k = 0
        for _ in range(self.MAX_REFINE_ITERS):
            state.polygon = polygon_from_bearings(state.bearings)
            if not state.polygon:
                return (False, None)
            center, radius = minimum_enclosing_circle(state.polygon)
            if radius <= CLEAR_RADIUS - 0.25:
                return (True, center)
            if self.time_left() < self.REFINE_RESERVE_S:
                break

            def fresh(p: Point) -> Point:
                for extra in range(0, 360, 30):
                    q = p if extra == 0 else self._perp_candidate(center, state, az_offset + extra)
                    if (q.x, q.y) not in visited:
                        return q
                return p
            if li < len(ladder) and radius > 100.0:
                candidate = fresh(ladder[li])
                li += 1
            elif no_signal_streak == 0:
                candidate = fresh(self._perp_candidate(center, state, az_offset))
            elif radius <= 150.0 and last_ok is not None:
                walk_k += 1
                dx, dy = (center.x - last_ok.x, center.y - last_ok.y)
                seg = math.hypot(dx, dy)
                if seg < 1e-06:
                    candidate = fresh(center)
                else:
                    nx, ny = (-dy / seg, dx / seg)
                    delta = max(30.0, min(120.0, 2.0 * radius))
                    side = (1.0, -1.0, 0.0)[(walk_k - 1) % 3]
                    base = Point(last_ok.x + 0.5 * dx, last_ok.y + 0.5 * dy)
                    candidate = fresh(Point(base.x + side * delta * nx, base.y + side * delta * ny))
            elif no_signal_streak % 2 == 1:
                candidate = fresh(Point((candidate.x + center.x) / 2, (candidate.y + center.y) / 2))
            else:
                az_offset += 60.0
                candidate = fresh(self._perp_candidate(center, state, az_offset))
            visited.append((candidate.x, candidate.y))
            self.current_pos = candidate
            response = self.api.measure(candidate, channel)
            require_accepted(response, f' 补测 (频道 {channel})')
            result = response.get('measure_result')
            if result == 'direction':
                state.bearings.append(Bearing(candidate, float(response['svd_deg'])))
                no_signal_streak = 0
                last_ok = candidate
            elif result == 'near':
                clear_response = self.api.clear(candidate, channel)
                require_accepted(clear_response, f' 清除 (频道 {channel})')
                state.cleared = clear_response.get('clear_result') == 'success'
                return (state.cleared, candidate if not state.cleared else None)
            else:
                no_signal_streak += 1
        state.polygon = polygon_from_bearings(state.bearings)
        if state.polygon:
            center, radius = minimum_enclosing_circle(state.polygon)
            if radius <= CLEAR_RADIUS - 0.25:
                response = self.api.clear(center, channel)
                require_accepted(response, f' 兜底清除 (频道 {channel})')
                if response.get('clear_result') == 'success':
                    state.cleared = True
                    return (True, None)
            elif self.time_left() >= FINE_GRID_RESERVE_S:
                if self.fine_grid_clear(channel):
                    return (True, None)
        return (False, None)

    def fine_grid_clear(self, channel: int) -> bool:
        state = self.states[channel]
        if not state.bearings or state.cleared:
            return False
        points = fine_grid_points(state.bearings)
        self.fine_grid_channels.append(channel)
        if not points:
            self.deferred_errors.append(f' 频道 {channel} 细网格兜底无候选点')
            return False
        for index, p in enumerate(points):
            if self.time_left() < FINE_GRID_RESERVE_S * 0.5:
                self.deferred_errors.append(f' 频道 {channel} 细网格兜底因时间保护中止 ({index}/{len(points)})')
                return False
            response = self.api.measure(p, channel)
            require_accepted(response, f' 细网格兜底 (频道 {channel})')
            result = response.get('measure_result')
            if result in ('direction', 'near'):
                clear_response = self.api.clear(p, channel)
                require_accepted(clear_response, f' 细网格兜底清除 (频道 {channel})')
                if clear_response.get('clear_result') == 'success':
                    state.cleared = True
                    return True
                self.deferred_errors.append(f' 频道 {channel} 细网格清除失败 (信号点)')
        return False

    @staticmethod
    def _perp_candidate(center: Point, state: ChannelState, az_offset: float) -> Point:
        last = state.bearings[-1]
        angle = math.radians(last.angle_deg + 90.0 + az_offset)
        step = 450.0
        candidates = [Point(center.x + step * math.cos(angle), center.y + step * math.sin(angle)), Point(center.x - step * math.cos(angle), center.y - step * math.sin(angle))]
        valid = [p for p in candidates if math.hypot(p.x, p.y) <= ARENA_RADIUS]
        if not valid:
            return center
        return min(valid, key=lambda p: p.distance(last.station))

    def run(self) -> dict:
        self.t0 = time.monotonic()
        entered = self.api.enter()
        if entered.get('accepted') is not True:
            raise RuntimeError(f'/enter 未被接受：{entered}')
        remaining = entered.get('remaining_real_duration_s')
        if isinstance(remaining, (int, float)) and remaining > 0:
            self.deadline = self.t0 + float(remaining)
        errors: list[str] = []
        self.current_pos = Point(0.0, 0.0)
        try:
            self.coverage_search()
        except (ActionRejected, RuntimeError) as exc:
            errors.append(f' 覆盖搜索中断：{exc}')
        current = self.current_pos
        ready: dict[int, Point] = dict(self.search_ready)
        failed: list[int] = []
        pending: dict[int, dict] = {}
        for channel, state in self.states.items():
            if state.cleared or not state.bearings or channel in ready:
                continue
            state.polygon = polygon_from_bearings(state.bearings)
            if not state.polygon:
                failed.append(channel)
                continue
            center, radius = minimum_enclosing_circle(state.polygon)
            if radius <= CLEAR_RADIUS - 0.25:
                ready[channel] = center
                continue
            pending[channel] = dict(li=0, attempts=0, no_signal_streak=0, az_offset=0.0, walk_k=0, kind='none', az_dedup=0.0, last_ok=state.bearings[-1].station, last_cand=None, visited=[], ladder=self._ladder_candidates(state))
        try:
            exhausted: list[int] = []
            while pending and self.time_left() >= self.REFINE_RESERVE_S:
                candidates: dict[int, Point] = {}
                finished: list[int] = []
                for channel, st in pending.items():
                    state = self.states[channel]
                    state.polygon = polygon_from_bearings(state.bearings)
                    if not state.polygon:
                        failed.append(channel)
                        finished.append(channel)
                        continue
                    center, radius = minimum_enclosing_circle(state.polygon)
                    if radius <= CLEAR_RADIUS - 0.25:
                        try:
                            clr = self.api.clear(center, channel)
                            require_accepted(clr, f' 就地清除 (频道 {channel})')
                            if clr.get('clear_result') == 'success':
                                state.cleared = True
                                self.search_ready.pop(channel, None)
                            else:
                                ready[channel] = center
                        except (ActionRejected, RuntimeError) as exc:
                            errors.append(f' 频道 {channel} 就地清除失败：{exc}')
                            ready[channel] = center
                        finished.append(channel)
                        continue
                    if st['attempts'] >= self.MAX_REFINE_ITERS:
                        finished.append(channel)
                        exhausted.append(channel)
                        continue
                    if st['li'] < len(st['ladder']) and radius > 100.0:
                        cand, kind = (st['ladder'][st['li']], 'ladder')
                    elif st['no_signal_streak'] == 0:
                        cand = self._perp_candidate(center, state, st['az_offset'])
                        kind = 'perp'
                    elif radius <= 150.0 and st['last_ok'] is not None:
                        dx, dy = (center.x - st['last_ok'].x, center.y - st['last_ok'].y)
                        seg = math.hypot(dx, dy)
                        if seg < 1e-06:
                            cand, kind = (center, 'none')
                        else:
                            nx, ny = (-dy / seg, dx / seg)
                            delta = max(30.0, min(120.0, 2.0 * radius))
                            side = (1.0, -1.0, 0.0)[st['walk_k'] % 3]
                            base = Point(st['last_ok'].x + 0.5 * dx, st['last_ok'].y + 0.5 * dy)
                            cand = Point(base.x + side * delta * nx, base.y + side * delta * ny)
                            kind = 'walk'
                    elif st['no_signal_streak'] % 2 == 1 and st['last_cand']:
                        cand = Point((st['last_cand'].x + center.x) / 2, (st['last_cand'].y + center.y) / 2)
                        kind = 'bisect'
                    else:
                        cand = self._perp_candidate(center, state, st['az_offset'] + 60.0)
                        kind = 'az'
                    st['kind'] = kind
                    st['az_dedup'] = st['az_offset'] + (60.0 if kind == 'az' else 0.0)
                    for extra in range(0, 360, 30):
                        q = cand if extra == 0 else self._perp_candidate(center, state, st['az_dedup'] + extra)
                        if (q.x, q.y) not in st['visited']:
                            cand = q
                            break
                    candidates[channel] = cand
                for channel in finished:
                    pending.pop(channel)
                if not candidates:
                    break
                channel = min(candidates, key=lambda c: current.distance(candidates[c]))
                cand = candidates[channel]
                st = pending[channel]
                if st['kind'] == 'ladder':
                    st['li'] += 1
                elif st['kind'] == 'walk':
                    st['walk_k'] += 1
                elif st['kind'] == 'az':
                    st['az_offset'] += 60.0
                st['visited'].append((cand.x, cand.y))
                st['last_cand'] = cand
                st['attempts'] += 1
                current = cand
                self.current_pos = cand
                response = self.api.measure(cand, channel)
                require_accepted(response, f' 补测 (频道 {channel})')
                result = response.get('measure_result')
                if result == 'direction':
                    st['last_ok'] = cand
                    self.states[channel].bearings.append(Bearing(cand, float(response['svd_deg'])))
                    st['no_signal_streak'] = 0
                elif result == 'near':
                    clear_response = self.api.clear(cand, channel)
                    require_accepted(clear_response, f' 清除 (频道 {channel})')
                    self.states[channel].cleared = clear_response.get('clear_result') == 'success'
                    if not self.states[channel].cleared:
                        failed.append(channel)
                    pending.pop(channel)
                else:
                    st['no_signal_streak'] += 1
        except (ActionRejected, RuntimeError) as exc:
            errors.append(f' 补测阶段中断：{exc}')
        for channel in list(dict.fromkeys(list(pending) + exhausted)):
            state = self.states[channel]
            state.polygon = polygon_from_bearings(state.bearings)
            if state.polygon:
                center, radius = minimum_enclosing_circle(state.polygon)
                if radius <= CLEAR_RADIUS - 0.25:
                    try:
                        response = self.api.clear(center, channel)
                        require_accepted(response, f' 兜底清除 (频道 {channel})')
                        if response.get('clear_result') == 'success':
                            state.cleared = True
                    except (ActionRejected, RuntimeError) as exc:
                        errors.append(f' 频道 {channel} 兜底失败：{exc}')
                elif self.time_left() >= FINE_GRID_RESERVE_S:
                    try:
                        self.fine_grid_clear(channel)
                    except (ActionRejected, RuntimeError) as exc:
                        errors.append(f' 频道 {channel} 细网格兜底失败：{exc}')
            if not state.cleared:
                failed.append(channel)
            pending.pop(channel, None)
        current = self.current_pos
        retry: list[int] = []
        clear_order = optimized_clear_order(current, ready)
        while clear_order and self.time_left() > self.CLEAR_RESERVE_S * 0.3:
            channel = clear_order.pop(0)
            clear_point = ready.pop(channel)
            try:
                response = self.api.clear(clear_point, channel)
                require_accepted(response, f' 清除 (频道 {channel})')
            except (ActionRejected, RuntimeError) as exc:
                errors.append(f' 频道 {channel} 清除失败：{exc}')
                failed.append(channel)
                continue
            current = clear_point
            if response.get('clear_result') == 'success':
                self.states[channel].cleared = True
            else:
                retry.append(channel)
        for channel in retry:
            if self.time_left() < self.CLEAR_RESERVE_S * 0.3:
                break
            try:
                ok, clear_point = self.refine(channel)
            except (ActionRejected, RuntimeError) as exc:
                errors.append(f' 频道 {channel} 回炉失败：{exc}')
                failed.append(channel)
                continue
            if ok and clear_point is not None:
                try:
                    response = self.api.clear(clear_point, channel)
                    require_accepted(response, f' 清除 (频道 {channel})')
                except (ActionRejected, RuntimeError) as exc:
                    errors.append(f' 频道 {channel} 清除失败：{exc}')
                    failed.append(channel)
                    continue
                current = clear_point
                if response.get('clear_result') == 'success':
                    self.states[channel].cleared = True
                else:
                    failed.append(channel)
            else:
                failed.append(channel)
        failed.extend(ready.keys())
        failed.extend(retry)
        errors.extend(self.deferred_errors)
        summary = {'discovered_channels': [c for c, s in self.states.items() if s.bearings or s.cleared], 'cleared_channels': [c for c, s in self.states.items() if s.cleared], 'failed_channels': sorted((c for c in set(failed) if not self.states[c].cleared)), 'unfound_channels': sorted((c for c, s in self.states.items() if not s.bearings and (not s.cleared))), 'fine_grid_fallback': sorted(self.fine_grid_channels), 'real_time_used_s': round(time.monotonic() - self.t0, 1), 'time_left_s': round(max(self.time_left(), 0.0), 1)}
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
    parser.add_argument('--log', default='robot-q4-actions.jsonl')
    args = parser.parse_args()
    if not args.live:
        print('dry-run：未连接模拟器。使用 --live --robot-id < 参赛队号> 才会执行测试。')
        return
    if not args.robot_id:
        raise SystemExit('live 模式必须提供 --robot-id')
    api = HttpRobotAPI(args.base_url, args.robot_id, Path(args.log))
    print(json.dumps(Q4Strategy(api).run(), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
