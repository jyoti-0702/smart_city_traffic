"""Load the city from CSV and apply time-of-day traffic."""
from __future__ import annotations

import csv
import os
import random
from typing import Dict, Optional, Set

from .graph import Edge, Graph, Node

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Multiplier applied to each road's base volume (base volume = typical busy-hour flow).
PERIODS: Dict[str, float] = {
    "Night (12am-5am)": 0.25,
    "Off-peak (11am-4pm)": 0.7,
    "Morning peak (8am-10am)": 1.2,
    "Evening peak (5pm-8pm)": 1.35,
}


def load_city(data_dir: Optional[str] = None, period: str = "Off-peak (11am-4pm)") -> Graph:
    data_dir = data_dir or DATA_DIR
    g = Graph()
    with open(os.path.join(data_dir, "nodes.csv"), newline="") as f:
        for r in csv.DictReader(f):
            g.add_node(Node(r["id"], r["name"], float(r["x_km"]), float(r["y_km"]), r["kind"]))
    with open(os.path.join(data_dir, "roads.csv"), newline="") as f:
        for r in csv.DictReader(f):
            e = Edge(int(r["id"]), r["from"], r["to"], float(r["distance_km"]),
                     float(r["speed_limit_kmh"]), float(r["capacity_vph"]),
                     float(r["base_volume_vph"]), r["road_type"])
            e.base_volume = float(r["base_volume_vph"])          # type: ignore[attr-defined]
            e.build_cost_per_km = float(r["build_cost_per_km"])  # type: ignore[attr-defined]
            g.add_edge(e)
    set_period(g, period)
    return g


def set_period(g: Graph, period: str) -> None:
    mult = PERIODS[period]
    for e in g.edges.values():
        e.volume = e.base_volume * mult  # type: ignore[attr-defined]


def add_incident(g: Graph, edge_id: int, extra_volume_factor: float = 0.6) -> None:
    """Simulate an accident / event that adds traffic to one road."""
    e = g.edges[edge_id]
    e.volume += e.capacity * extra_volume_factor


def build_cost(e: Edge) -> float:
    """Relative construction cost of a road (cost units per km x km)."""
    return e.distance_km * getattr(e, "build_cost_per_km", 1.0)


def name(g: Graph, node_id: str) -> str:
    return g.nodes[node_id].name


def route_names(g: Graph, nodes) -> str:
    return " -> ".join(g.nodes[n].name for n in nodes)


def random_trips(g: Graph, n: int, seed: int = 7):
    rng = random.Random(seed)
    ids = list(g.nodes)
    trips = []
    while len(trips) < n:
        a, b = rng.sample(ids, 2)
        trips.append((a, b))
    return trips
