"""Tests: every algorithm is compared with networkx on the sample city and on random graphs."""
import math
import os
import random
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smartcity import city
from smartcity.benchmark import random_city
from smartcity.graph import (Edge, Graph, Node, alternative_routes, astar, bellman_ford, bfs,
                             bridges, connected_components, dfs, dijkstra, kruskal, prim,
                             w_distance, w_time)


def to_nx(g, weight=w_time, closed=()):
    G = nx.Graph()
    G.add_nodes_from(g.nodes)
    for e in g.edges.values():
        if e.id not in closed:
            G.add_edge(e.u, e.v, w=weight(e), eid=e.id)
    return G


@pytest.fixture(scope="module")
def sample():
    return city.load_city(period="Evening peak (5pm-8pm)")


def graphs():
    yield "sample", city.load_city(period="Morning peak (8am-10am)")
    for seed in (1, 2, 3):
        yield f"random{seed}", random_city(60, seed=seed)


@pytest.mark.parametrize("weight", [w_time, w_distance])
def test_shortest_paths_match_networkx(weight):
    for label, g in graphs():
        G = to_nx(g, weight)
        rng = random.Random(5)
        ids = list(g.nodes)
        for _ in range(25):
            a, b = rng.sample(ids, 2)
            want = nx.dijkstra_path_length(G, a, b, "w")
            for fn in (dijkstra, bellman_ford, astar):
                got = fn(g, a, b, weight)
                assert got.found
                assert got.cost == pytest.approx(want), (label, fn.__name__, a, b)
                # the returned path must be a real walk whose edge costs add up
                assert got.nodes[0] == a and got.nodes[-1] == b
                total = sum(weight(g.edges[e]) for e in got.edge_ids)
                assert total == pytest.approx(got.cost)


def test_astar_explores_no_more_than_dijkstra(sample):
    worse = 0
    rng = random.Random(3)
    ids = list(sample.nodes)
    for _ in range(40):
        a, b = rng.sample(ids, 2)
        if astar(sample, a, b).explored > dijkstra(sample, a, b).explored:
            worse += 1
    assert worse <= 4   # heuristic should almost never explore more


def test_kruskal_and_prim_match_networkx():
    for label, g in graphs():
        G = to_nx(g, w_distance)
        want = sum(d["w"] for _, _, d in nx.minimum_spanning_edges(G, weight="w", data=True))
        k_ids, k_total = kruskal(g)
        p_ids, p_total = prim(g)
        assert k_total == pytest.approx(want), label
        assert p_total == pytest.approx(want), label
        assert len(k_ids) == len(g.nodes) - 1
        assert len(p_ids) == len(g.nodes) - 1
        # spanning: the chosen edges connect every junction
        T = nx.Graph()
        T.add_nodes_from(g.nodes)
        T.add_edges_from((g.edges[i].u, g.edges[i].v) for i in k_ids)
        assert nx.is_connected(T)


def test_bfs_dfs_components():
    for label, g in graphs():
        G = to_nx(g)
        src = next(iter(g.nodes))
        order, hops = bfs(g, src)
        assert hops == nx.single_source_shortest_path_length(G, src)
        assert set(dfs(g, src)) == set(order)
        assert len(dfs(g, src)) == len(set(dfs(g, src)))        # no repeats
        got = sorted(sorted(c) for c in connected_components(g))
        want = sorted(sorted(c) for c in nx.connected_components(G))
        assert got == want


def test_bridges_match_networkx():
    for label, g in graphs():
        G = to_nx(g)
        want = {frozenset(e) for e in nx.bridges(G)}
        got = {frozenset((g.edges[i].u, g.edges[i].v)) for i in bridges(g)}
        assert got == want, label


def test_closed_roads_change_route_and_can_disconnect(sample):
    best = dijkstra(sample, "N01", "N30")
    closed = {best.edge_ids[0]}
    detour = dijkstra(sample, "N01", "N30", closed=closed)
    assert detour.found and detour.cost >= best.cost - 1e-9
    assert closed.isdisjoint(detour.edge_ids)
    # closing a bridge cuts the dead-end village off
    village_road = next(i for i in bridges(sample)
                        if "N31" in (sample.edges[i].u, sample.edges[i].v))
    cut = dijkstra(sample, "N31", "N30", closed={village_road})
    assert not cut.found
    assert not bellman_ford(sample, "N31", "N30", closed={village_road}).found
    assert not astar(sample, "N31", "N30", closed={village_road}).found
    assert len(connected_components(sample, {village_road})) == 2


def test_alternative_routes(sample):
    routes = alternative_routes(sample, "N01", "N30", k=3)
    assert 2 <= len(routes) <= 3
    assert len({tuple(r.edge_ids) for r in routes}) == len(routes)       # all different
    assert routes[0].cost == pytest.approx(dijkstra(sample, "N01", "N30").cost)
    assert [r.cost for r in routes] == sorted(r.cost for r in routes)


def test_traffic_model_bpr():
    e = Edge(1, "a", "b", distance_km=10, speed_limit=60, capacity=1000, volume=0)
    assert e.free_flow_min == pytest.approx(10.0)
    assert e.travel_min == pytest.approx(10.0)
    e.volume = 1000                                   # v/c = 1  -> 15% slower
    assert e.travel_min == pytest.approx(11.5)
    e.volume = 2000                                   # v/c = 2  -> 1 + 0.15 * 16 = 3.4x
    assert e.travel_min == pytest.approx(34.0)


def test_peak_traffic_slows_trips():
    night = city.load_city(period="Night (12am-5am)")
    evening = city.load_city(period="Evening peak (5pm-8pm)")
    assert dijkstra(evening, "N01", "N30").cost > dijkstra(night, "N01", "N30").cost


def test_bad_input():
    g = Graph()
    g.add_node(Node("a", "A", 0, 0))
    with pytest.raises(KeyError):
        g.add_edge(Edge(1, "a", "zzz", 1, 50, 100))
    g.add_node(Node("b", "B", 1, 0))
    assert not dijkstra(g, "a", "b").found      # two nodes, no road
    assert dijkstra(g, "a", "a").cost == 0
