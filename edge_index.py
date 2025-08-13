from typing import List, Tuple, Optional, Dict
import math

import numpy as np
from scipy.spatial import KDTree


def _dot3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def segment_distance_sq(
    a0: Tuple[float, float, float],
    a1: Tuple[float, float, float],
    b0: Tuple[float, float, float],
    b1: Tuple[float, float, float],
) -> float:
    u = (a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2])
    v = (b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2])
    w0 = (a0[0] - b0[0], a0[1] - b0[1], a0[2] - b0[2])
    a = _dot3(u, u)
    b = _dot3(u, v)
    c = _dot3(v, v)
    d = _dot3(u, w0)
    e = _dot3(v, w0)
    denom = a * c - b * b
    if denom < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        sD = denom
        tD = denom
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
            sD = 1.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
            sD = 1.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    dx = w0[0] + sc * u[0] - tc * v[0]
    dy = w0[1] + sc * u[1] - tc * v[1]
    dz = w0[2] + sc * u[2] - tc * v[2]
    return dx * dx + dy * dy + dz * dz


class EdgeIndex:
    """Spatial index for inter-tree edge proximity checks.
    Uses a KDTree over edge midpoints for O(log N) candidate lookup, then exact segment distance checks.
    """

    def __init__(self) -> None:
        self.p0_list: List[Tuple[float, float, float]] = []
        self.p1_list: List[Tuple[float, float, float]] = []
        self.midpoints: List[Tuple[float, float, float]] = []
        self.half_lengths: List[float] = []
        self.tree_ids: List[int] = []
        self.kdtree: Optional[KDTree] = None
        self._dirty: bool = False
        self._adds_since_rebuild: int = 0
        self.tree_min_distance: Dict[int, float] = {}

    def set_tree_min_distance(self, tree_id: int, min_dist: float) -> None:
        self.tree_min_distance[int(tree_id)] = float(min_dist)

    def _rebuild_if_needed(self, force: bool = False) -> None:
        if self._dirty and (force or self._adds_since_rebuild >= 128):
            if self.midpoints:
                self.kdtree = KDTree(np.array(self.midpoints))
            else:
                self.kdtree = None
            self._dirty = False
            self._adds_since_rebuild = 0

    def rebuild(self) -> None:
        self._rebuild_if_needed(force=True)

    def add_edge(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float], tree_id: int) -> None:
        p0 = (float(p0[0]), float(p0[1]), float(p0[2]))
        p1 = (float(p1[0]), float(p1[1]), float(p1[2]))
        self.p0_list.append(p0)
        self.p1_list.append(p1)
        mx = (p0[0] + p1[0]) * 0.5
        my = (p0[1] + p1[1]) * 0.5
        mz = (p0[2] + p1[2]) * 0.5
        self.midpoints.append((mx, my, mz))
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        self.half_lengths.append(0.5 * math.sqrt(dx * dx + dy * dy + dz * dz))
        self.tree_ids.append(int(tree_id))
        self._dirty = True
        self._adds_since_rebuild += 1
        self._rebuild_if_needed()

    def validate_edge(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float], tree_id: int) -> bool:
        self._rebuild_if_needed(force=False)
        min_dist = float(self.tree_min_distance.get(int(tree_id), 0.0))
        if self.kdtree is None or min_dist <= 0.0:
            return True
        mx = (p0[0] + p1[0]) * 0.5
        my = (p0[1] + p1[1]) * 0.5
        mz = (p0[2] + p1[2]) * 0.5
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        half_len = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
        radius = min_dist + half_len
        try:
            idxs = self.kdtree.query_ball_point([mx, my, mz], r=radius)
        except Exception:
            return True
        for idx in idxs:
            if self.tree_ids[idx] == int(tree_id):
                continue
            omx, omy, omz = self.midpoints[idx]
            ddx = mx - omx
            ddy = my - omy
            ddz = mz - omz
            other_half = self.half_lengths[idx]
            if ddx * ddx + ddy * ddy + ddz * ddz > (radius + other_half) * (radius + other_half):
                continue
            if segment_distance_sq(p0, p1, self.p0_list[idx], self.p1_list[idx]) < (min_dist * min_dist):
                return False
        return True


def _dot3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def segment_distance_sq(
    a0: Tuple[float, float, float],
    a1: Tuple[float, float, float],
    b0: Tuple[float, float, float],
    b1: Tuple[float, float, float],
) -> float:
    u = (a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2])
    v = (b1[0] - b0[0], b1[1] - b0[1], b1[2] - b0[2])
    w0 = (a0[0] - b0[0], a0[1] - b0[1], a0[2] - b0[2])
    a = _dot3(u, u)
    b = _dot3(u, v)
    c = _dot3(v, v)
    d = _dot3(u, w0)
    e = _dot3(v, w0)
    denom = a * c - b * b
    if denom < 1e-12:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        sD = denom
        tD = denom
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
            sD = 1.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
            sD = 1.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    dx = w0[0] + sc * u[0] - tc * v[0]
    dy = w0[1] + sc * u[1] - tc * v[1]
    dz = w0[2] + sc * u[2] - tc * v[2]
    return dx * dx + dy * dy + dz * dz


class EdgeIndex:
    """Spatial index for inter-tree edge proximity checks.
    Uses a KDTree over edge midpoints for O(log N) candidate lookup, then exact segment distance checks.
    """

    def __init__(self) -> None:
        self.p0_list: List[Tuple[float, float, float]] = []
        self.p1_list: List[Tuple[float, float, float]] = []
        self.midpoints: List[Tuple[float, float, float]] = []
        self.half_lengths: List[float] = []
        self.tree_ids: List[int] = []
        self.kdtree: Optional[KDTree] = None
        self._dirty: bool = False
        self._adds_since_rebuild: int = 0
        self.tree_min_distance: Dict[int, float] = {}

    def set_tree_min_distance(self, tree_id: int, min_dist: float) -> None:
        self.tree_min_distance[int(tree_id)] = float(min_dist)

    def _rebuild_if_needed(self, force: bool = False) -> None:
        if self._dirty and (force or self._adds_since_rebuild >= 128):
            if self.midpoints:
                self.kdtree = KDTree(np.array(self.midpoints))
            else:
                self.kdtree = None
            self._dirty = False
            self._adds_since_rebuild = 0

    def rebuild(self) -> None:
        self._rebuild_if_needed(force=True)

    def add_edge(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float], tree_id: int) -> None:
        p0 = (float(p0[0]), float(p0[1]), float(p0[2]))
        p1 = (float(p1[0]), float(p1[1]), float(p1[2]))
        self.p0_list.append(p0)
        self.p1_list.append(p1)
        mx = (p0[0] + p1[0]) * 0.5
        my = (p0[1] + p1[1]) * 0.5
        mz = (p0[2] + p1[2]) * 0.5
        self.midpoints.append((mx, my, mz))
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        self.half_lengths.append(0.5 * math.sqrt(dx * dx + dy * dy + dz * dz))
        self.tree_ids.append(int(tree_id))
        self._dirty = True
        self._adds_since_rebuild += 1
        self._rebuild_if_needed()

    def validate_edge(self, p0: Tuple[float, float, float], p1: Tuple[float, float, float], tree_id: int) -> bool:
        self._rebuild_if_needed(force=False)
        min_dist = float(self.tree_min_distance.get(int(tree_id), 0.0))
        if self.kdtree is None or min_dist <= 0.0:
            return True
        mx = (p0[0] + p1[0]) * 0.5
        my = (p0[1] + p1[1]) * 0.5
        mz = (p0[2] + p1[2]) * 0.5
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        half_len = 0.5 * math.sqrt(dx * dx + dy * dy + dz * dz)
        radius = min_dist + half_len
        try:
            idxs = self.kdtree.query_ball_point([mx, my, mz], r=radius)
        except Exception:
            return True
        for idx in idxs:
            if self.tree_ids[idx] == int(tree_id):
                continue
            omx, omy, omz = self.midpoints[idx]
            ddx = mx - omx
            ddy = my - omy
            ddz = mz - omz
            other_half = self.half_lengths[idx]
            if ddx * ddx + ddy * ddy + ddz * ddz > (radius + other_half) * (radius + other_half):
                continue
            if segment_distance_sq(p0, p1, self.p0_list[idx], self.p1_list[idx]) < (min_dist * min_dist):
                return False
        return True


