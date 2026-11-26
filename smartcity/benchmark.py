"""Performance analysis: run every algorithm on road networks of growing size."""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Dict, List

from .graph import (Edge, Graph, Node, astar, bellman_ford, bfs, bridges, dfs, dijkstra,
                    kruskal, prim, w_distance, w_time)

SIZES = [100, 250, 500, 1000, 2000, 4000]


def random_city(n: int, seed: int = 0, extra: float = 0.6) -> Graph:
    """Connected random road network with n junctions (grid-like, planar-ish).

    Junctions are scattered in a square; each connects to a few nearest
    neighbours, then a spanning chain guarantees connectivity.
    """
    rng = random.Random(seed)
    side = math.sqrt(n) * 1.0
    g = Graph()
    pts = []
    for i in range(n):
        x, y = rng.uniform(0, side), rng.uniform(0, side)
        g.add_node(Node(f"R{i}", f"R{i}", x, y))
        pts.append((x, y, f"R{i}"))
    # grid buckets for fast nearest-neighbour search
    cell = 1.5
    buckets: Dict[tuple, list] = {}
    for p in pts:
        buckets.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(p)
    pair_set = set()

    def link(a, b):
        key = tuple(sorted((a[2], b[2])))
        if key in pair_set or a[2] == b[2]:
            return
        pair_set.add(key)
        d = math.dist((a[0], a[1]), (b[0], b[1])) * rng.uniform(1.02, 1.2)
        typ = rng.choice([(40, 900), (40, 900), (60, 1800), (80, 3200)])
        cap = typ[1]
        eid = len(g.edges) + 1
        e = Edge(eid, a[2], b[2], max(d, 0.05), typ[0], cap, cap * rng.uniform(0.2, 1.1))
        e.base_volume = e.volume          # type: ignore[attr-defined]
        g.add_edge(e)

    for p in pts:
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        near = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near += buckets.get((cx + dx, cy + dy), [])
        near.sort(key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)
        for q in near[1:3 + int(extra * 3)]:
            link(p, q)
    # guarantee connectivity: chain components through their closest pairs
    order = sorted(pts, key=lambda q: (q[0], q[1]))
    from .graph import connected_components
    comps = connected_components(g)
    while len(comps) > 1:
        base = set(comps[0])
        best = None
        for p in pts:
            if p[2] in base:
                for q in pts:
                    if q[2] not in base:
                        d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, p, q)
        link(best[1], best[2])
        comps = connected_components(g)
    return g


def _time(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0


def run(sizes: List[int] = SIZES, trips: int = 5, seed: int = 1) -> List[dict]:
    rows = []
    for n in sizes:
        g = random_city(n, seed=seed)
        rng = random.Random(seed)
        ids = list(g.nodes)
        pairs = [tuple(rng.sample(ids, 2)) for _ in range(trips)]
        src = ids[0]
        reps = 3 if n <= 1000 else 1
        row = {"junctions": len(g.nodes), "roads": len(g.edges)}
        row["BFS"] = _time(lambda: bfs(g, src), reps)
        row["DFS"] = _time(lambda: dfs(g, src), reps)
        row["Dijkstra"] = _time(lambda: [dijkstra(g, a, b) for a, b in pairs], reps) / trips
        row["A*"] = _time(lambda: [astar(g, a, b) for a, b in pairs], reps) / trips
        bf_pairs = pairs[:2] if n > 1000 else pairs
        row["Bellman-Ford"] = _time(lambda: [bellman_ford(g, a, b) for a, b in bf_pairs], 1) / len(bf_pairs)
        row["Kruskal"] = _time(lambda: kruskal(g), reps)
        row["Prim"] = _time(lambda: prim(g), reps)
        row["Bridges"] = _time(lambda: bridges(g), reps)
        row["Dijkstra explored"] = statistics.mean(dijkstra(g, a, b).explored for a, b in pairs)
        row["A* explored"] = statistics.mean(astar(g, a, b).explored for a, b in pairs)
        rows.append(row)
    return rows


if __name__ == "__main__":
    import json
    import sys
    sizes = SIZES if len(sys.argv) < 2 else [int(s) for s in sys.argv[1:]]
    out = run(sizes)
    for r in out:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    with open("docs/benchmark_results.json", "w") as f:
        json.dump(out, f, indent=1)
