"""Plotly figures for the road network."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import plotly.graph_objects as go

from .graph import Graph

BINS = [  # (upper bound of volume/capacity, colour, label)
    (0.6, "#2e9d4f", "Free flow (<60%)"),
    (0.9, "#f2b600", "Busy (60-90%)"),
    (1.1, "#e8590c", "Heavy (90-110%)"),
    (99, "#b00020", "Jammed (>110%)"),
]
ROUTE_COLOURS = ["#1565c0", "#7b1fa2", "#00838f"]


def _line(g: Graph, eids: Iterable[int]):
    xs, ys = [], []
    for eid in eids:
        e = g.edges[eid]
        a, b = g.nodes[e.u], g.nodes[e.v]
        xs += [a.x, b.x, None]
        ys += [a.y, b.y, None]
    return xs, ys


def network_figure(g: Graph, routes: Optional[Sequence[Sequence[int]]] = None,
                   route_labels: Optional[Sequence[str]] = None,
                   closed: Optional[set] = None, mst: Optional[Iterable[int]] = None,
                   critical: Optional[Iterable[int]] = None,
                   endpoints: Optional[Sequence[str]] = None,
                   show_traffic: bool = True, height: int = 560,
                   visited: Optional[Sequence[str]] = None, title: str = "") -> go.Figure:
    fig = go.Figure()
    closed = closed or set()
    live = [eid for eid in g.edges if eid not in closed]
    if show_traffic:
        lo = 0.0
        for hi, colour, label in BINS:
            ids = [i for i in live if lo <= g.edges[i].congestion < hi]
            lo = hi
            if ids:
                xs, ys = _line(g, ids)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label,
                                         line=dict(color=colour, width=3), hoverinfo="skip"))
    else:
        xs, ys = _line(g, live)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Roads",
                                 line=dict(color="#9aa5b1", width=2), hoverinfo="skip"))
    if closed:
        xs, ys = _line(g, closed)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Closed",
                                 line=dict(color="#111", width=3, dash="dot"), hoverinfo="skip"))
    if mst:
        xs, ys = _line(g, mst)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Minimum spanning tree",
                                 line=dict(color="#1565c0", width=6), opacity=0.55, hoverinfo="skip"))
    if critical:
        xs, ys = _line(g, critical)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Critical road (bridge)",
                                 line=dict(color="#d81b60", width=7), opacity=0.6, hoverinfo="skip"))
    for i, r in enumerate(routes or []):
        xs, ys = _line(g, r)
        label = route_labels[i] if route_labels else f"Route {i + 1}"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label,
                                 line=dict(color=ROUTE_COLOURS[i % 3], width=8 - 2 * i), opacity=0.9,
                                 hoverinfo="skip"))
    # hover points at road midpoints
    mx, my, tip = [], [], []
    for e in g.edges.values():
        a, b = g.nodes[e.u], g.nodes[e.v]
        mx.append((a.x + b.x) / 2); my.append((a.y + b.y) / 2)
        tip.append(f"Road {e.id}: {a.name} - {b.name}<br>{e.road_type}, {e.distance_km:.2f} km<br>"
                   f"load {e.congestion * 100:.0f}% | {e.travel_min:.1f} min (free flow {e.free_flow_min:.1f})")
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=9, opacity=0),
                             hovertext=tip, hoverinfo="text", showlegend=False))
    vset = set(visited or [])
    ends = set(endpoints or [])
    colours = ["#d81b60" if n in ends else ("#1565c0" if n in vset else "#263238") for n in g.nodes]
    sizes = [15 if n in ends else (12 if n in vset else 9) for n in g.nodes]
    fig.add_trace(go.Scatter(
        x=[n.x for n in g.nodes.values()], y=[n.y for n in g.nodes.values()], mode="markers+text",
        text=[n.name if (n.id in ends or len(g.nodes) <= 40) else "" for n in g.nodes.values()],
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=sizes, color=colours, line=dict(color="white", width=1)),
        hovertext=[f"{n.name} ({n.id}, {n.kind})" for n in g.nodes.values()], hoverinfo="text",
        name="Junctions", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=title, plot_bgcolor="#f7f9fb",
                      xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.02, font=dict(size=11)))
    return fig
