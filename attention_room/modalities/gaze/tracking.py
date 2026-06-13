"""
Identify person
"""

import math
from typing import List, Tuple

class FaceTracker:
    def __init__(self, gate: float = 0.12, max_age: int = 15) -> None:
        self.gate = gate
        self.max_age = max_age
        self._next_id = 0
        self._tracks: dict[int, dict] = {} # id -> {cx, cy, age}

    def update(self, centroids: List[Tuple[float, float]]) -> List[int]:
        ids: List[int] = [-1] * len(centroids)

        pairs = [] # (distance, det_index, track_id)
        for di, (cx, cy) in enumerate(centroids):
            for tid, t in self._tracks.items():
                d = math.hypot(cx - t["cx"], cy - t["cy"])
                if d <= self.gate:
                    pairs.append((d, di, tid))
        
        pairs.sort()

        used_det, used_trk = set(), set()
        for d, di, tid in pairs:
            if di in used_det or tid in used_trk:
                continue
            used_det.add(di)
            used_trk.add(tid)
            ids[di] = tid
            self._tracks[tid].update(cx=centroids[di][0], cy=centroids[di][1], age=0)

        new_ids = set()
        for di, (cx, cy) in enumerate(centroids):
            if ids[di] == -1:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"cx": cx, "cy": cy, "age": 0}
                ids[di] = tid
                new_ids.add(tid)
        
        for tid in list(self._tracks):
            if tid not in used_trk and tid not in new_ids:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        
        return ids
