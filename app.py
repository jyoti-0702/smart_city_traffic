"""Smart City Traffic Management and Route Optimization System (Streamlit app).

Run:  streamlit run app.py
"""
import json
import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from smartcity import benchmark, city
from smartcity.graph import (alternative_routes, astar, bellman_ford, bfs, bridges,
                             connected_components, dfs, dijkstra, edge_usage, kruskal, prim,
                             w_distance, w_free_flow, w_time)
from smartcity.viz import network_figure

st.set_page_config(page_title="Smart City Traffic System", page_icon="🚦", layout="wide")

ALGOS = {"Dijkstra": dijkstra, "A*": astar, "Bellman-Ford": bellman_ford}
st.title("🚦 Smart City Traffic Management & Route Optimization")
st.caption("A synthetic city modelled as a weighted graph: junctions are nodes, roads are edges whose "
           "travel time grows with congestion (BPR function).")

with st.sidebar:
    st.header("Traffic conditions")
    period = st.selectbox("Time of day", list(city.PERIODS), index=3)
    g = city.load_city(period=period)
    names = {n.id: n.name for n in g.nodes.values()}
    ids = list(g.nodes)
    st.subheader("Road closures")
    closed_ids = st.multiselect(
        "Close roads", sorted(g.edges),
        format_func=lambda i: f"#{i} {names[g.edges[i].u]} - {names[g.edges[i].v]}")
    closed = set(closed_ids)
    st.subheader("Incident")
    inc = st.selectbox("Add an incident on a road", [None] + sorted(g.edges),
                       format_func=lambda i: "None" if i is None else
                       f"#{i} {names[g.edges[i].u]} - {names[g.edges[i].v]}")
    if inc is not None:
        city.add_incident(g, inc, 0.6)
    st.caption(f"{len(g.nodes)} junctions · {len(g.edges)} roads · {len(closed)} closed")

tab_commuter, tab_admin, tab_explore, tab_perf = st.tabs(
    ["🧭 Commuter route planner", "🏛 City administrator", "🔍 Network explorer", "⏱ Performance analysis"])

# ----------------------------------------------------------------- commuter
with tab_commuter:
    c1, c2, c3, c4 = st.columns([2, 2, 1.4, 1.4])
    src = c1.selectbox("From", ids, index=ids.index("N01"), format_func=lambda i: names[i])
    dst = c2.selectbox("To", ids, index=ids.index("N30"), format_func=lambda i: names[i])
    goal = c3.radio("Optimise for", ["Fastest (with traffic)", "Shortest distance"])
    algo_name = c4.selectbox("Algorithm", list(ALGOS))
    weight = w_time if goal.startswith("Fastest") else w_distance
    if src == dst:
        st.info("Pick two different junctions.")
    else:
        algo = ALGOS[algo_name]
        t0 = time.perf_counter()
        best = algo(g, src, dst, weight, closed)
        elapsed = (time.perf_counter() - t0) * 1000
        if not best.found:
            st.error("No route exists: the closed roads have cut these two junctions apart.")
        else:
            routes = alternative_routes(g, src, dst, 3, weight, closed)
            dist_km = sum(g.edges[e].distance_km for e in best.edge_ids)
            mins = sum(g.edges[e].travel_min for e in best.edge_ids)
            free = sum(g.edges[e].free_flow_min for e in best.edge_ids)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Travel time", f"{mins:.1f} min")
            m2.metric("Distance", f"{dist_km:.2f} km")
            m3.metric("Delay from traffic", f"{mins - free:.1f} min",
                      help=f"Free-flow time would be {free:.1f} min")
            m4.metric(f"{algo_name} ran in", f"{elapsed:.2f} ms",
                      help=f"{best.explored} junctions/relaxations examined")
            labels = []
            for i, r in enumerate(routes):
                rm = sum(g.edges[e].travel_min for e in r.edge_ids)
                rd = sum(g.edges[e].distance_km for e in r.edge_ids)
                labels.append(f"{'Best route' if i == 0 else f'Alternative {i}'} ({rm:.0f} min, {rd:.1f} km)")
            left, right = st.columns([3, 2])
            left.plotly_chart(network_figure(g, [r.edge_ids for r in routes], labels, closed,
                                             endpoints=[src, dst]), width="stretch")
            with right:
                st.markdown("**Turn-by-turn (best route)**")
                for i, eid in enumerate(best.edge_ids):
                    e = g.edges[eid]
                    st.write(f"{i + 1}. {names[best.nodes[i]]} → {names[best.nodes[i + 1]]} "
                             f"· {e.road_type} · {e.travel_min:.1f} min · load {e.congestion * 100:.0f}%")
                rows = []
                for i, r in enumerate(routes):
                    rows.append({"Route": "Best" if i == 0 else f"Alt {i}",
                                 "Minutes": round(sum(g.edges[e].travel_min for e in r.edge_ids), 1),
                                 "Km": round(sum(g.edges[e].distance_km for e in r.edge_ids), 2),
                                 "Roads": len(r.edge_ids),
                                 "Worst road load %": round(max(g.edges[e].congestion for e in r.edge_ids) * 100)})
                st.markdown("**Route comparison**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# -------------------------------------------------------------------- admin
with tab_admin:
    st.subheader("Traffic status")
    top = sorted(g.edges.values(), key=lambda e: e.congestion, reverse=True)
    jammed = sum(e.congestion > 1.1 for e in g.edges.values())
    heavy = sum(0.9 < e.congestion <= 1.1 for e in g.edges.values())
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Average road load", f"{sum(e.congestion for e in g.edges.values()) / len(g.edges) * 100:.0f}%")
    a2.metric("Jammed roads", jammed)
    a3.metric("Heavy roads", heavy)
    comps = connected_components(g, closed)
    a4.metric("Separate zones", len(comps), help="More than 1 means closures have split the city.")
    col_map, col_tbl = st.columns([3, 2])
    mode = st.radio("Map layer", ["Traffic", "Critical roads (bridges)", "Minimum spanning tree"],
                    horizontal=True)
    crit = bridges(g, closed) if mode.startswith("Critical") else None
    mst_ids = kruskal(g, city.build_cost, closed)[0] if mode.startswith("Minimum") else None
    col_map.plotly_chart(network_figure(g, closed=closed, critical=crit, mst=mst_ids),
                         width="stretch")
    with col_tbl:
        st.markdown("**Most congested roads**")
        st.dataframe(pd.DataFrame([{"Road": f"#{e.id} {names[e.u]} - {names[e.v]}", "Type": e.road_type,
                                    "Load %": round(e.congestion * 100), "Minutes": round(e.travel_min, 1),
                                    "Delay min": round(e.travel_min - e.free_flow_min, 1)}
                                   for e in top[:10]]), hide_index=True, width="stretch")
        if crit is not None:
            st.markdown(f"**{len(crit)} critical roads.** Closing any one of these disconnects part of the "
                        "city, so they need a backup route or priority maintenance.")
            st.write(", ".join(f"#{i} ({names[g.edges[i].u]} - {names[g.edges[i].v]})" for i in crit))

    st.divider()
    st.subheader("Infrastructure planning: minimum spanning tree")
    kr, kr_cost = kruskal(g, city.build_cost, closed)
    pr, pr_cost = prim(g, city.build_cost, closed=closed)
    full_cost = sum(city.build_cost(e) for i, e in g.edges.items() if i not in closed)
    p1, p2, p3 = st.columns(3)
    p1.metric("Backbone roads (MST)", f"{len(kr)} of {len(g.edges) - len(closed)}")
    p2.metric("Backbone build cost", f"{kr_cost:.1f} units",
              help="Kruskal and Prim give the same total: " + f"{pr_cost:.1f}")
    p3.metric("Whole network cost", f"{full_cost:.1f} units")
    st.caption("The MST is the cheapest set of roads that still connects every junction. Roads outside it "
               "are redundancy. The congested backbone roads below are the first candidates for widening.")
    cong_backbone = sorted((g.edges[i] for i in kr), key=lambda e: e.congestion, reverse=True)[:5]
    st.dataframe(pd.DataFrame([{"Backbone road": f"#{e.id} {names[e.u]} - {names[e.v]}",
                                "Load %": round(e.congestion * 100), "Type": e.road_type,
                                "Build cost": round(city.build_cost(e), 1)} for e in cong_backbone]),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Impact of closures on citywide trips")
    base = city.load_city(period=period)
    if inc is not None:
        city.add_incident(base, inc, 0.6)
    trips = city.random_trips(base, 40)
    before = [dijkstra(base, a, b) for a, b in trips]
    after = [dijkstra(g, a, b, closed=closed) for a, b in trips]
    ok = [(x, y) for x, y in zip(before, after) if y.found]
    cut = len(trips) - len(ok)
    if ok:
        avg_b = sum(x.cost for x, _ in ok) / len(ok)
        avg_a = sum(y.cost for _, y in ok) / len(ok)
        i1, i2, i3 = st.columns(3)
        i1.metric("Average trip (no closures)", f"{avg_b:.1f} min")
        i2.metric("Average trip (with closures)", f"{avg_a:.1f} min", f"{avg_a - avg_b:+.1f} min",
                  delta_color="inverse")
        i3.metric("Trips with no route", f"{cut} of {len(trips)}")
    st.caption("Based on 40 fixed sample trips between random junctions.")
    usage = edge_usage(base, trips)
    busiest = sorted(usage.items(), key=lambda kv: kv[1], reverse=True)[:5]
    st.markdown("**Roads used by the most trips (bottleneck candidates):** " + ", ".join(
        f"#{i} {names[base.edges[i].u]}-{names[base.edges[i].v]} ({n} trips)" for i, n in busiest))

# ----------------------------------------------------------------- explorer
with tab_explore:
    e1, e2 = st.columns([1, 3])
    with e1:
        start = st.selectbox("Start junction", ids, format_func=lambda i: names[i], key="exp")
        method = st.radio("Traversal", ["BFS", "DFS"])
        hops_limit = st.slider("Max hops (BFS)", 1, 8, 3)
        order, hops = bfs(g, start, closed)
        if method == "BFS":
            reach = [n for n in order if hops[n] <= hops_limit]
            st.write(f"**{len(reach)}** junctions within {hops_limit} hops.")
            show = reach
        else:
            show = dfs(g, start, closed)
            st.write(f"DFS reaches **{len(show)}** of {len(g.nodes)} junctions.")
        st.markdown("**Visit order**")
        st.write(" → ".join(names[n] for n in show[:15]) + (" ..." if len(show) > 15 else ""))
        unreachable = [names[n] for n in g.nodes if n not in set(order)]
        if unreachable:
            st.warning("Cut off by closures: " + ", ".join(unreachable))
    e2.plotly_chart(network_figure(g, closed=closed, show_traffic=False, visited=show,
                                   endpoints=[start]), width="stretch")

# -------------------------------------------------------------- performance
with tab_perf:
    st.write("Time taken by each algorithm on randomly generated road networks of growing size "
             "(median of repeated runs; Dijkstra, A* and Bellman-Ford are per route).")
    path = os.path.join(os.path.dirname(__file__), "docs", "benchmark_results.json")
    if st.button("Re-run benchmark live (up to 1000 junctions)"):
        with st.spinner("Running..."):
            rows = benchmark.run([100, 250, 500, 1000])
    elif os.path.exists(path):
        rows = json.load(open(path))
    else:
        rows = benchmark.run([100, 250, 500, 1000])
    df = pd.DataFrame(rows)
    algos = ["BFS", "DFS", "Dijkstra", "A*", "Bellman-Ford", "Kruskal", "Prim", "Bridges"]
    long = df.melt(id_vars=["junctions"], value_vars=algos, var_name="Algorithm", value_name="ms")
    fig = px.line(long, x="junctions", y="ms", color="Algorithm", markers=True, log_x=True, log_y=True,
                  labels={"ms": "time (ms)", "junctions": "junctions"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    p1, p2 = st.columns([3, 2])
    p1.plotly_chart(fig, width="stretch")
    ex = df[["junctions", "Dijkstra explored", "A* explored"]].round(0)
    p2.markdown("**Junctions examined per route**")
    p2.dataframe(ex, hide_index=True, width="stretch")
    p2.caption("A* examines fewer junctions than Dijkstra because its straight-line estimate steers it "
               "toward the destination.")
    st.dataframe(df[["junctions", "roads"] + algos].round(3), hide_index=True, width="stretch")
