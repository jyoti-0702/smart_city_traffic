"""Road network graph and the graph algorithms used by the system.

Everything here is implemented from scratch (no networkx) so the algorithms
can be explained and measured. networkx is used only in the tests, as an
independent check.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

INF = float("inf")


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    x: float          # km, east-west
    y: float          # km, north-south
    kind: str = "junction"


@dataclass
class Edge:
    id: int
    u: str
    v: str
    distance_km: float
    speed_limit: float        # km/h
    capacity: float           # vehicles per hour
    volume: float = 0.0       # vehicles per hour right now
    road_type: str = "street"

    def other(self, node: str) -> str:
        return self.v if node == self.u else self.u

    @property
    def free_flow_min(self) -> float:
        return self.distance_km / self.speed_limit * 60.0

    @property
    def congestion(self) -> float:
        """Volume / capacity ratio (1.0 = road is full)."""
        return self.volume / self.capacity if self.capacity else 0.0

    @property
    def travel_min(self) -> float:
        """Travel time with congestion, BPR function: t0 * (1 + 0.15 * (v/c)^4)."""
        return self.free_flow_min * (1.0 + 0.15 * self.congestion ** 4)


WeightFn = Callable[[Edge], float]


def w_distance(e: Edge) -> float:
    return e.distance_km


def w_time(e: Edge) -> float:
    return e.travel_min


def w_free_flow(e: Edge) -> float:
    return e.free_flow_min


class Graph:
    """Undirected weighted multigraph-free road network (adjacency list)."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.adj: Dict[str, List[Tuple[str, int]]] = {}

    # -- construction -----------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adj.setdefault(node.id, [])

    def add_edge(self, edge: Edge) -> None:
        if edge.u not in self.nodes or edge.v not in self.nodes:
            raise KeyError(f"unknown node in edge {edge.id}")
        self.edges[edge.id] = edge
        self.adj[edge.u].append((edge.v, edge.id))
        self.adj[edge.v].append((edge.u, edge.id))

    def __len__(self) -> int:
        return len(self.nodes)

    def neighbors(self, node: str, closed: Optional[Set[int]] = None):
        for nbr, eid in self.adj[node]:
            if closed and eid in closed:
                continue
            yield nbr, eid

    def euclid_km(self, a: str, b: str) -> float:
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(na.x - nb.x, na.y - nb.y)

    def max_speed(self) -> float:
        return max(e.speed_limit for e in self.edges.values())


# ---------------------------------------------------------------- traversal
def bfs(g: Graph, source: str, closed: Optional[Set[int]] = None):
    """Breadth-first search. Returns (visit order, hop distance per node)."""
    hops = {source: 0}
    order = [source]
    q = deque([source])
    while q:
        cur = q.popleft()
        for nbr, _ in g.neighbors(cur, closed):
            if nbr not in hops:
                hops[nbr] = hops[cur] + 1
                order.append(nbr)
                q.append(nbr)
    return order, hops


def dfs(g: Graph, source: str, closed: Optional[Set[int]] = None) -> List[str]:
    """Iterative depth-first search. Returns visit order."""
    seen, order, stack = set(), [], [source]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        order.append(cur)
        for nbr, _ in reversed(list(g.neighbors(cur, closed))):
            if nbr not in seen:
                stack.append(nbr)
    return order


def connected_components(g: Graph, closed: Optional[Set[int]] = None) -> List[List[str]]:
    seen: Set[str] = set()
    comps = []
    for n in g.nodes:
        if n not in seen:
            comp = dfs(g, n, closed)
            seen.update(comp)
            comps.append(comp)
    return comps


def reachable_within(g: Graph, source: str, max_hops: int, closed=None) -> List[str]:
    _, hops = bfs(g, source, closed)
    return [n for n, h in hops.items() if h <= max_hops]


# ------------------------------------------------------------ shortest path
@dataclass
class PathResult:
    found: bool
    cost: float = INF
    nodes: Optional[List[str]] = None
    edge_ids: Optional[List[int]] = None
    explored: int = 0          # nodes taken off the priority queue / relaxed


def _rebuild(prev: dict, src: str, dst: str):
    nodes, eids = [dst], []
    cur = dst
    while cur != src:
        p, eid = prev[cur]
        eids.append(eid)
        nodes.append(p)
        cur = p
    nodes.reverse(); eids.reverse()
    return nodes, eids


def dijkstra(g: Graph, src: str, dst: str, weight: WeightFn = w_time,
             closed: Optional[Set[int]] = None) -> PathResult:
    dist = {src: 0.0}
    prev: dict = {}
    pq = [(0.0, src)]
    done: Set[str] = set()
    while pq:
        d, cur = heapq.heappop(pq)
        if cur in done:
            continue
        done.add(cur)
        if cur == dst:
            nodes, eids = _rebuild(prev, src, dst)
            return PathResult(True, d, nodes, eids, len(done))
        for nbr, eid in g.neighbors(cur, closed):
            nd = d + weight(g.edges[eid])
            if nd < dist.get(nbr, INF):
                dist[nbr] = nd
                prev[nbr] = (cur, eid)
                heapq.heappush(pq, (nd, nbr))
    return PathResult(False, INF, None, None, len(done))


def dijkstra_all(g: Graph, src: str, weight: WeightFn = w_time, closed=None):
    """Single-source distances and predecessor map to every node."""
    dist = {src: 0.0}
    prev: dict = {}
    pq = [(0.0, src)]
    done: Set[str] = set()
    while pq:
        d, cur = heapq.heappop(pq)
        if cur in done:
            continue
        done.add(cur)
        for nbr, eid in g.neighbors(cur, closed):
            nd = d + weight(g.edges[eid])
            if nd < dist.get(nbr, INF):
                dist[nbr] = nd
                prev[nbr] = (cur, eid)
                heapq.heappush(pq, (nd, nbr))
    return dist, prev


def bellman_ford(g: Graph, src: str, dst: str, weight: WeightFn = w_time,
                 closed: Optional[Set[int]] = None) -> PathResult:
    """Bellman-Ford with early exit. Roads are undirected, so each is relaxed both ways."""
    dist = {n: INF for n in g.nodes}
    dist[src] = 0.0
    prev: dict = {}
    relaxations = 0
    live = [e for e in g.edges.values() if not (closed and e.id in closed)]
    for _ in range(len(g.nodes) - 1):
        changed = False
        for e in live:
            w = weight(e)
            for a, b in ((e.u, e.v), (e.v, e.u)):
                relaxations += 1
                if dist[a] + w < dist[b]:
                    dist[b] = dist[a] + w
                    prev[b] = (a, e.id)
                    changed = True
        if not changed:
            break
    if dist[dst] == INF:
        return PathResult(False, INF, None, None, relaxations)
    nodes, eids = _rebuild(prev, src, dst)
    return PathResult(True, dist[dst], nodes, eids, relaxations)


def astar(g: Graph, src: str, dst: str, weight: WeightFn = w_time,
          closed: Optional[Set[int]] = None, heuristic_scale: Optional[float] = None) -> PathResult:
    """A* with a straight-line heuristic.

    `heuristic_scale` converts km to the weight unit: 1.0 for distance (km),
    60 / max_speed for travel time (minutes at the fastest speed limit). Both
    never overestimate, so the result is optimal.
    """
    if heuristic_scale is None:
        heuristic_scale = 1.0 if weight is w_distance else 60.0 / g.max_speed()
    h = lambda n: g.euclid_km(n, dst) * heuristic_scale
    gscore = {src: 0.0}
    prev: dict = {}
    pq = [(h(src), src)]
    done: Set[str] = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in done:
            continue
        done.add(cur)
        if cur == dst:
            nodes, eids = _rebuild(prev, src, dst)
            return PathResult(True, gscore[cur], nodes, eids, len(done))
        for nbr, eid in g.neighbors(cur, closed):
            ng = gscore[cur] + weight(g.edges[eid])
            if ng < gscore.get(nbr, INF):
                gscore[nbr] = ng
                prev[nbr] = (cur, eid)
                heapq.heappush(pq, (ng + h(nbr), nbr))
    return PathResult(False, INF, None, None, len(done))


def alternative_routes(g: Graph, src: str, dst: str, k: int = 3, weight: WeightFn = w_time,
                       closed: Optional[Set[int]] = None, penalty: float = 1.6) -> List[PathResult]:
    """Up to k distinct routes using the penalty method.

    After each route is found, its roads get a temporary cost multiplier so the
    next search prefers different roads. Reported costs use the real weights.
    """
    extra: Dict[int, float] = {}
    routes: List[PathResult] = []
    seen: Set[Tuple[int, ...]] = set()
    for _ in range(k * 4):
        wfn = lambda e: weight(e) * extra.get(e.id, 1.0)
        r = dijkstra(g, src, dst, wfn, closed)
        if not r.found:
            break
        key = tuple(r.edge_ids)
        for eid in r.edge_ids:
            extra[eid] = extra.get(eid, 1.0) * penalty
        if key in seen:
            continue
        seen.add(key)
        r.cost = sum(weight(g.edges[e]) for e in r.edge_ids)
        routes.append(r)
        if len(routes) == k:
            break
    routes.sort(key=lambda r: r.cost)
    return routes


# -------------------------------------------------- minimum spanning tree
class DisjointSet:
    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal(g: Graph, weight: WeightFn = w_distance, closed=None) -> Tuple[List[int], float]:
    """Minimum spanning forest. Returns (edge ids, total weight)."""
    ds = DisjointSet(g.nodes)
    chosen, total = [], 0.0
    for e in sorted((e for e in g.edges.values() if not (closed and e.id in closed)), key=weight):
        if ds.union(e.u, e.v):
            chosen.append(e.id)
            total += weight(e)
    return chosen, total


def prim(g: Graph, weight: WeightFn = w_distance, start: Optional[str] = None,
         closed=None) -> Tuple[List[int], float]:
    """Minimum spanning tree of the component containing `start` (heap-based Prim)."""
    start = start or next(iter(g.nodes))
    seen = {start}
    pq = [(weight(g.edges[eid]), eid, nbr) for nbr, eid in g.neighbors(start, closed)]
    heapq.heapify(pq)
    chosen, total = [], 0.0
    while pq:
        w, eid, node = heapq.heappop(pq)
        if node in seen:
            continue
        seen.add(node)
        chosen.append(eid)
        total += w
        for nbr, e2 in g.neighbors(node, closed):
            if nbr not in seen:
                heapq.heappush(pq, (weight(g.edges[e2]), e2, nbr))
    return chosen, total


# ------------------------------------------------ resilience / bottlenecks
def bridges(g: Graph, closed: Optional[Set[int]] = None) -> List[int]:
    """Edges whose closure disconnects the network (iterative Tarjan low-link)."""
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    out: List[int] = []
    t = 0
    for root in g.nodes:
        if root in disc:
            continue
        disc[root] = low[root] = t; t += 1
        stack = [(root, -1, iter(list(g.neighbors(root, closed))))]
        while stack:
            node, pe, it = stack[-1]
            advanced = False
            for nbr, eid in it:
                if eid == pe:
                    continue
                if nbr in disc:
                    low[node] = min(low[node], disc[nbr])
                else:
                    disc[nbr] = low[nbr] = t; t += 1
                    stack.append((nbr, eid, iter(list(g.neighbors(nbr, closed)))))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > disc[parent]:
                        out.append(pe)
    return out


def edge_usage(g: Graph, pairs: List[Tuple[str, str]], weight: WeightFn = w_time,
               closed=None) -> Dict[int, int]:
    """How many of the given trips use each road on their fastest route."""
    usage: Dict[int, int] = {eid: 0 for eid in g.edges}
    for a, b in pairs:
        r = dijkstra(g, a, b, weight, closed)
        if r.found:
            for eid in r.edge_ids:
                usage[eid] += 1
    return usage
