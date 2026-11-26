"""Build docs/Capstone_Report.pdf from the real code, test output, benchmark and screenshots."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from smartcity import city
from smartcity.graph import (alternative_routes, astar, bellman_ford, bridges, connected_components,
                             dijkstra, kruskal, prim, w_distance)

OUT = "docs/Capstone_Report.pdf"
SS = "docs/screenshots"
FIG = "docs/figures"
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------ real numbers
g = city.load_city(period="Evening peak (5pm-8pm)")
names = {n.id: n.name for n in g.nodes.values()}
best = dijkstra(g, "N01", "N30")
free = sum(g.edges[e].free_flow_min for e in best.edge_ids)
closed = {48, 36}
detour = dijkstra(g, "N01", "N30", closed=closed)
alts = alternative_routes(g, "N01", "N30", 3)
kr, kr_cost = kruskal(g, city.build_cost)
_, pr_cost = prim(g, city.build_cost)
full_cost = sum(city.build_cost(e) for e in g.edges.values())
crit = bridges(g)
night = city.load_city(period="Night (12am-5am)")
cong = sorted(g.edges.values(), key=lambda e: e.congestion, reverse=True)
avg_load = sum(e.congestion for e in g.edges.values()) / len(g.edges) * 100
trips = city.random_trips(g, 40)
b4 = [dijkstra(g, a, b) for a, b in trips]
af = [dijkstra(g, a, b, closed=closed) for a, b in trips]
ok = [(x, y) for x, y in zip(b4, af) if y.found]
avg_b = sum(x.cost for x, _ in ok) / len(ok)
avg_a = sum(y.cost for _, y in ok) / len(ok)
bench = json.load(open("docs/benchmark_results.json"))
b1000 = next(r for r in bench if r["junctions"] == 1000)
b4000 = next(r for r in bench if r["junctions"] == 4000)
same3 = all(abs(f(g, "N01", "N30").cost - best.cost) < 1e-9 for f in (astar, bellman_ford))
tests_line = open("docs/pytest_output.txt").read().strip().splitlines()[-1].strip("= ")
print("facts:", round(best.cost, 2), round(detour.cost, 2), len(kr), round(kr_cost, 1), round(full_cost, 1), len(crit))

# ----------------------------------------------------------- figures
def arch_figure():
    fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)

    def box(x, y, w, h, text, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#37474f", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#37474f"))
    box(0.1, 1.7, 1.8, 1.0, "data/nodes.csv\ndata/roads.csv\n(sample city)", "#fff3e0")
    box(2.4, 1.7, 1.9, 1.0, "city.py\nloader, time-of-day\ntraffic, incidents", "#e3f2fd")
    box(4.8, 1.7, 2.0, 1.0, "graph.py\nGraph + algorithms", "#e8f5e9")
    box(7.3, 3.0, 2.5, 1.0, "app.py (Streamlit)\nCommuter | Admin |\nExplorer | Performance", "#f3e5f5")
    box(7.3, 1.7, 2.5, 1.0, "viz.py\nPlotly network map", "#f3e5f5")
    box(7.3, 0.4, 2.5, 1.0, "benchmark.py\nrandom cities, timing", "#fce4ec")
    box(4.8, 0.2, 2.0, 0.9, "tests/test_graph.py\nchecked vs networkx", "#fffde7")
    for a, b in [((1.9, 2.2), (2.4, 2.2)), ((4.3, 2.2), (4.8, 2.2)), ((6.8, 2.4), (7.3, 3.4)),
                 ((6.8, 2.2), (7.3, 2.2)), ((6.8, 2.0), (7.3, 0.9)), ((5.8, 1.7), (5.8, 1.1))]:
        arrow(*a, *b)
    ax.text(5, 4.15, "System architecture", ha="center", fontsize=11, weight="bold")
    fig.savefig(f"{FIG}/architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def bench_figure():
    algos = ["BFS", "DFS", "Dijkstra", "A*", "Bellman-Ford", "Kruskal", "Prim", "Bridges"]
    xs = [r["junctions"] for r in bench]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a in algos:
        ax[0].plot(xs, [r[a] for r in bench], marker="o", label=a)
    ax[0].set(xscale="log", yscale="log", xlabel="junctions", ylabel="time (ms)", title="Run time vs network size")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
    w = 0.38; x = np.arange(len(xs))
    ax[1].bar(x - w / 2, [r["Dijkstra explored"] for r in bench], w, label="Dijkstra")
    ax[1].bar(x + w / 2, [r["A* explored"] for r in bench], w, label="A*")
    ax[1].set_xticks(x, xs); ax[1].set(xlabel="junctions", ylabel="junctions examined per route",
                                       title="Search effort: Dijkstra vs A*")
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{FIG}/benchmark.png", dpi=170); plt.close(fig)


arch_figure(); bench_figure()


def trimmed(path, top=0, bottom=None, pad=18):
    """Crop a screenshot: optional top/bottom, then trim empty white space at the bottom."""
    im = PILImage.open(path).convert("RGB")
    arr = np.array(im)
    bottom = bottom or arr.shape[0]
    arr = arr[top:bottom, 312:1460]          # drop the sidebar and the scrollbar
    rows = np.where((arr < 245).any(axis=(1, 2)))[0]
    end = min(arr.shape[0], rows.max() + pad)
    out = path.replace(".png", "_crop.png").replace(SS, FIG)
    PILImage.fromarray(arr[:end]).save(out)
    return out


# ----------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceBefore=10, textColor=colors.HexColor("#1f3a5f"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, textColor=colors.HexColor("#1f3a5f"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=7.8, leading=9.6)
W = 17 * cm


def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [P("&bull; " + i) for i in items]


def table(data, widths, font=8.5):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")])]))
    return t


def shot(path, caption, max_h=17.5 * cm):
    im = PILImage.open(path); w, h = im.size
    height = W * h / w
    width = W
    if height > max_h:
        width, height = W * max_h / height, max_h
    return KeepTogether([Image(path, width=width, height=height), P(caption, SM), Spacer(1, 6)])


def wrapped(cells, widths, font=8.5):
    st = ParagraphStyle("c", parent=B, fontSize=font, leading=font + 2.5)
    return table([[Paragraph(str(c), st) if i else Paragraph(f"<b>{c}</b>", st) for i, c in enumerate(r)] if False
                  else [Paragraph(str(c), st) for c in r] for r in cells], widths, font)


# ------------------------------------------------------------------ story
s = []
s += [Spacer(1, 3 * cm), P("Smart City Traffic Management<br/>and Route Optimization System",
                           ParagraphStyle("T", parent=ss["Title"], fontSize=24, leading=30)),
      Spacer(1, 10), P("DAA Lab 4 - Capstone Project Report", ParagraphStyle("S", parent=ss["Title"], fontSize=14)),
      Spacer(1, 1.2 * cm)]
info = [["Name", "Bulbul"], ["Roll No.", "2401201214"], ["Specialisation", "BCA (AI & Data Science)"],
        ["Section", "A"], ["Course", "Design and Analysis of Algorithms (ENCA301)"],
        ["Instructor", "Aarti Sangwan"], ["University", "K.R. Mangalam University"],
        ["GitHub repository", "Data not added"]]
t = Table(info, colWidths=[4.5 * cm, 10 * cm])
t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 11),
                       ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
s += [t, Spacer(1, 1.5 * cm),
      P("Note: all traffic and road data in this project is synthetic. It was generated by "
        "scripts/generate_city.py for a fictional city and is not real traffic data.", SM), PageBreak()]

s += [P("1. Problem statement", H1),
      P("Cities lose time and money to congestion, and commuters usually only see the road in front of them. "
        "The goal is a Smart City Traffic Management and Route Optimization System that models the road "
        "network as a graph and uses graph algorithms to support routing and traffic-management decisions."),
      P("Objectives", H2)] + bullets([
    "Model junctions as nodes and roads as weighted edges with distance, speed limit, capacity and live traffic volume.",
    "Implement graph traversal (BFS, DFS), shortest path (Dijkstra, A*, Bellman-Ford) and minimum spanning tree (Kruskal, Prim).",
    "Analyse the network: find optimal and alternative routes, evaluate traffic, find critical roads, plan infrastructure.",
    "Give commuters a route planner and city administrators a decision-support dashboard, with interactive visualisations.",
    "Measure performance on growing networks and verify correctness against an independent library."])

s += [P("2. System design", H1), Image(f"{FIG}/architecture.png", width=W, height=W * 0.44),
      P("Figure 1. Architecture. Data flows left to right; the tests and benchmark use the same graph library.", SM),
      P("2.1 Data model", H2),
      wrapped([["Entity", "Fields"],
               ["Junction (node)", "id, name, x/y position in km, kind (transit, hospital, airport, ...)"],
               ["Road (edge)", "id, two junctions, distance_km, speed_limit_kmh, capacity_vph, base_volume_vph, road_type, build_cost_per_km"],
               ["Network (graph)", "adjacency list: junction -> list of (neighbour, road id). Roads are two-way."]],
              [3.5 * cm, 13.5 * cm]),
      P("The sample city has 32 junctions and 51 roads in three types (street, arterial, expressway). It has a "
        "river with only two crossings and two dead-end villages, so the network has realistic weak points."),
      P("2.2 Traffic model", H2),
      P("Travel time on a road uses the Bureau of Public Roads (BPR) function: "
        "<b>t = t0 x (1 + 0.15 x (v / c)^4)</b>, where t0 is free-flow time, v is the current volume and c is "
        "the capacity. A road at capacity is about 15% slower; at twice capacity it is 3.4 times slower. "
        "Four time-of-day periods scale each road's base volume (night 0.25, off-peak 0.7, morning peak 1.2, "
        "evening peak 1.35). Incidents add extra volume to one road."),
      P("2.3 Which algorithm answers which question", H2),
      wrapped([["Question", "Algorithm", "Time complexity"],
               ["Fastest route from A to B", "Dijkstra with a binary heap", "O((V + E) log V)"],
               ["Same, with fewer junctions examined", "A* (straight-line heuristic)", "O((V + E) log V) worst case"],
               ["Cross-check, and a method that tolerates negative weights", "Bellman-Ford", "O(V x E)"],
               ["Alternative routes", "Dijkstra repeated with a penalty on used roads", "k x Dijkstra"],
               ["What can I reach in N hops? Is the city still connected?", "BFS, DFS, connected components", "O(V + E)"],
               ["Cheapest road backbone that connects every junction", "Kruskal (union-find) and Prim (heap)", "O(E log E) / O(E log V)"],
               ["Which roads, if closed, split the city?", "Bridge finding (Tarjan low-link, iterative)", "O(V + E)"],
               ["Which roads carry the most trips?", "Dijkstra for many trips, count road use", "trips x Dijkstra"]],
              [6.2 * cm, 6.3 * cm, 4.5 * cm])]

s += [P("3. Implementation", H1),
      P("The project is written in Python 3. All graph algorithms are implemented from scratch in "
        "<font face='Courier'>smartcity/graph.py</font> (about 400 lines) so they can be explained and measured. "
        "networkx is used only in the tests as an independent check."),
      wrapped([["File", "Purpose"],
               ["smartcity/graph.py", "Graph class; BFS, DFS, components; Dijkstra, A*, Bellman-Ford; alternative routes; Kruskal, Prim; bridges; edge usage"],
               ["smartcity/city.py", "Loads CSV data, applies time-of-day traffic, adds incidents, computes road build cost"],
               ["smartcity/viz.py", "Plotly map coloured by congestion; overlays for routes, closures, MST and critical roads"],
               ["smartcity/benchmark.py", "Random connected road networks and timing of every algorithm"],
               ["app.py", "Streamlit app with four tabs"],
               ["tests/test_graph.py", "11 tests comparing every algorithm with networkx"],
               ["scripts/", "Sample-city generator, screenshot capture, this report builder"]],
              [4.2 * cm, 12.8 * cm]),
      P("Design decisions", H2)] + bullets([
    "<b>One weight function per query.</b> Each algorithm takes a function (road -> cost), so the same code "
    "answers 'shortest distance', 'fastest with traffic' and 'free-flow time'.",
    "<b>Closed roads are a parameter.</b> Every traversal and search accepts a set of closed road ids, so "
    "closures, detours and 'cut-off zone' checks reuse the same algorithms.",
    "<b>A* stays optimal.</b> Its straight-line estimate never overestimates: road lengths are at least the "
    "straight-line distance, and the time estimate uses the fastest speed limit in the network.",
    "<b>Iterative, not recursive.</b> DFS and the bridge search use explicit stacks, so large networks do not hit Python's recursion limit.",
    "<b>Alternative routes by penalty.</b> After each route is found its roads get a cost multiplier of 1.6, "
    "so the next search prefers different roads. Reported costs use the real, unpenalised times."])

s += [P("4. Features and results", H1),
      P("All screenshots below were captured from the running application (Streamlit, Plotly) with "
        "scripts/take_screenshots.py. Map colours: green = free flow (under 60% of capacity), "
        "yellow = busy, orange = heavy, red = jammed (over 110%).")]
s += [P("4.1 Commuter route planner", H2),
      P(f"For Northgate to Sports Complex at the evening peak the best route takes "
        f"<b>{best.cost:.1f} min</b> over <b>{sum(g.edges[e].distance_km for e in best.edge_ids):.2f} km</b>, "
        f"{best.cost - free:.1f} min of it traffic delay (free flow would be {free:.1f} min). Dijkstra, A* and "
        f"Bellman-Ford return the same cost ({'verified' if same3 else 'MISMATCH'}). The planner also offers "
        f"{len(alts) - 1} alternatives ({', '.join(f'{r.cost:.1f} min' for r in alts[1:])}) and a turn-by-turn list."),
      shot(trimmed(f"{SS}/01_commuter_route.png"), "Figure 2. Commuter route planner: best route (thick blue) and two alternatives, with route comparison table.")]
s += [P("4.2 Road closures and incidents", H2),
      P(f"Closing two roads on that route (Tech Park - Civil Lines and Civil Lines - Airport Road) forces a "
        f"detour: the trip rises from {best.cost:.1f} to <b>{detour.cost:.1f} min</b>. Across 40 fixed sample trips "
        f"the average trip goes from {avg_b:.1f} to {avg_a:.1f} min ({avg_a - avg_b:+.1f} min) and "
        f"{len(trips) - len(ok)} of {len(trips)} trips lose their route entirely."),
      shot(trimmed(f"{SS}/07_closure_commuter.png"), "Figure 3. Two closed roads (dotted black); the planner re-routes automatically.")]
s += [P("4.3 City administrator dashboard", H2),
      P(f"At the evening peak the average road load is {avg_load:.0f}%. The most congested road is "
        f"#{cong[0].id} {names[cong[0].u]} - {names[cong[0].v]} at {cong[0].congestion * 100:.0f}% of capacity. "
        f"The dashboard lists the busiest roads, shows how many separate zones the city is split into, and "
        f"ranks roads by how many sample trips depend on them (bottleneck candidates)."),
      shot(trimmed(f"{SS}/02_admin_traffic.png", bottom=1500), "Figure 4. Administrator dashboard: traffic status, congested roads and closure impact.", 19 * cm)]
s += [P("4.4 Critical roads (bridge finding)", H2),
      P(f"The bridge algorithm finds <b>{len(crit)} critical roads</b> in the sample city: "
        + ", ".join(f"#{i} ({names[g.edges[i].u]} - {names[g.edges[i].v]})" for i in crit)
        + ". Closing any one of them cuts off a district: Lakeside Village, Quarry Hamlet, or the Westend and Hillcrest spur. "
          "These need a backup route or priority maintenance."),
      shot(trimmed(f"{SS}/03_admin_critical_roads.png", top=300, bottom=1100), "Figure 5. Critical roads highlighted in pink.")]
s += [P("4.5 Infrastructure planning with minimum spanning trees", H2),
      P(f"Using construction cost (cost-per-km x length) as the weight, Kruskal and Prim both find a backbone "
        f"of <b>{len(kr)} of {len(g.edges)} roads</b> costing <b>{kr_cost:.1f}</b> units, against "
        f"{full_cost:.1f} units for the whole network ({(1 - kr_cost / full_cost) * 100:.0f}% less). "
        f"Both algorithms return the same total ({kr_cost:.1f} vs {pr_cost:.1f}). Roads outside the backbone are "
        f"redundancy. Backbone roads that are already congested are the first candidates for widening."),
      shot(trimmed(f"{SS}/04_admin_mst.png", top=300, bottom=1700), "Figure 6. Minimum spanning tree (blue overlay) and congested backbone roads.", 19 * cm)]
s += [P("4.6 Network explorer (BFS and DFS)", H2),
      P("Pick a junction and a hop limit to see which junctions BFS reaches, or the DFS visit order. "
        "If closures cut part of the city off, the explorer lists the unreachable junctions."),
      shot(trimmed(f"{SS}/05_explorer_bfs.png"), "Figure 7. BFS from a junction: reached junctions in blue.", 14 * cm)]

s += [P("5. Testing", H1),
      P(f"The test suite has 11 tests and all pass ({tests_line}). Each algorithm is compared with networkx on the "
        "sample city under several traffic periods and on three random 60-junction networks, using 25 random "
        "junction pairs for each weight type.")]
s += [wrapped([["Test", "What it checks"],
               ["Shortest paths vs networkx", "Dijkstra, Bellman-Ford and A* give the optimal cost for distance and for travel time; returned path's edge costs add up to the reported cost"],
               ["A* search effort", "A* examines no more junctions than Dijkstra in nearly all trips"],
               ["Kruskal and Prim vs networkx", "Both give the networkx MST weight and a spanning tree with V-1 edges that connects all junctions"],
               ["BFS, DFS, components", "BFS hop counts equal networkx; DFS reaches the same set without repeats; components match"],
               ["Bridges vs networkx", "The critical-road set equals networkx's bridges"],
               ["Closures", "Closed roads are never used; a detour is no faster; closing a bridge makes the route unreachable in all three algorithms and splits the city into two components"],
               ["Alternative routes", "Routes are distinct, sorted by cost, and the first equals the Dijkstra optimum"],
               ["Traffic model", "BPR gives 10.0, 11.5 and 34.0 min at v/c of 0, 1 and 2; evening peak is slower than night"],
               ["Bad input", "Unknown junction raises an error; disconnected junctions return 'no route'; source = destination costs 0"]],
              [4.8 * cm, 12.2 * cm]),
      Spacer(1, 6), P("Test output", H2),
      Preformatted(open("docs/pytest_output.txt").read(), CODE), Spacer(1, 4),
      P("The app itself was also exercised with Streamlit's AppTest harness (no exceptions with closures set "
        "and time period changed) and with a real browser to produce the screenshots in this report.", SM)]

s += [P("6. Performance analysis", H1),
      P("Each algorithm was timed on random connected road networks from 100 to 4,000 junctions "
        "(median of repeated runs; route algorithms are averaged per route; Bellman-Ford on fewer routes at "
        "large sizes). Times are from the machine that built this report, so absolute values will differ on "
        "yours; the growth trends are what matter."),
      Image(f"{FIG}/benchmark.png", width=W, height=W * 0.38),
      P("Figure 8. Left: run time against network size (log-log). Right: junctions examined per route.", SM)]
rows = [["Junctions", "Roads", "BFS", "DFS", "Dijkstra", "A*", "Bellman-Ford", "Kruskal", "Prim", "Bridges"]]
for r in bench:
    rows.append([r["junctions"], r["roads"]] + [f"{r[k]:.2f}" for k in
                 ["BFS", "DFS", "Dijkstra", "A*", "Bellman-Ford", "Kruskal", "Prim", "Bridges"]])
s += [table(rows, [1.8 * cm, 1.4 * cm] + [1.5 * cm] * 4 + [2.3 * cm] + [1.5 * cm] * 3, 8), P("Table 1. Time in milliseconds.", SM),
      P("Observations", H2)] + bullets([
    f"Bellman-Ford is far slower: at 1,000 junctions it takes {b1000['Bellman-Ford']:.1f} ms per route against "
    f"{b1000['Dijkstra']:.2f} ms for Dijkstra. At 4,000 junctions it is {b4000['Bellman-Ford']:.0f} ms against "
    f"{b4000['Dijkstra']:.1f} ms. Its O(V x E) cost is only worth paying when negative weights are possible, which road travel times never are.",
    f"A* examines fewer junctions than Dijkstra, {b1000['A* explored']:.0f} against {b1000['Dijkstra explored']:.0f} "
    f"at 1,000 junctions ({(1 - b1000['A* explored'] / b1000['Dijkstra explored']) * 100:.0f}% fewer), and "
    f"{b4000['A* explored']:.0f} against {b4000['Dijkstra explored']:.0f} at 4,000. Wall-clock times are close, though: "
    "A* is not always faster at small sizes, because each junction also costs a distance calculation for the heuristic.",
    "BFS, DFS, Kruskal, Prim, Dijkstra and the bridge search all grow roughly in proportion to network size "
    "(about 40-50 times slower for 40 times more junctions), consistent with O((V + E) log V) or better.",
    "Even the slowest everyday operation (a route query) finishes in a few milliseconds at 4,000 junctions, "
    "so the app can recompute routes instantly when a commuter changes a closure or a time period."])

s += [P("7. Conclusions", H1)] + bullets([
    "The system meets the capstone brief: it models a city as a graph and uses traversal, shortest-path and "
    "minimum-spanning-tree algorithms for routing, traffic evaluation and infrastructure planning, with an "
    "interactive app for commuters and administrators.",
    f"Traffic matters: at the evening peak the sample route carries {best.cost - free:.1f} min of delay, and two "
    f"closures nearly double the trip time ({best.cost:.1f} to {detour.cost:.1f} min).",
    f"Planning insight: a backbone of {len(kr)} roads connects the whole city for {(1 - kr_cost / full_cost) * 100:.0f}% "
    f"less build cost, and {len(crit)} critical roads are single points of failure.",
    "Dijkstra is the right default for routing; A* saves search effort on large maps; Bellman-Ford is only a cross-check here.",
    "Every algorithm was verified against networkx, which gives confidence that the numbers shown to users are correct."])
s += [P("Limitations and future work", H2)] + bullets([
    "The city and traffic are synthetic, with one static snapshot per time period. A real deployment would read a live traffic feed.",
    "Roads are two-way and turn costs are ignored. One-way streets would need a directed graph.",
    "Alternative routes use a penalty heuristic, not Yen's k-shortest-paths algorithm.",
    "Possible extensions: real OpenStreetMap data, signal timing, public-transport layers, route prediction from history."])
s += [P("8. How to run", H1),
      Preformatted("pip install -r requirements.txt\nstreamlit run app.py          # the application\n"
                   "python -m pytest -q tests     # 11 tests\npython -m smartcity.benchmark # performance analysis", CODE),
      P("Source code, README with execution steps, sample data and logs: GitHub link - Data not added.", SM)]


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.1 * cm, "Smart City Traffic Management - Bulbul, 2401201214, Section A")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Smart City Traffic Management - Capstone Report",
                  author="Bulbul").build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
