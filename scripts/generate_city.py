"""Generate the synthetic sample city (data/nodes.csv, data/roads.csv).

The city is fictional. Run:  python scripts/generate_city.py
"""
import csv
import math
import os
import random

SEED = 2026
COLS, ROWS, SPACING = 6, 5, 1.3
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

NAMES = [
    "Northgate", "Maple Court", "Riverside", "Old Market", "Lake View", "Eastfield",
    "Parkside", "Tech Park", "Central Station", "City Hospital", "Mall Road", "Harbor Point",
    "Green Avenue", "University", "Town Hall", "Clock Tower", "Sunrise Colony", "Stadium",
    "Westend", "Industrial Area", "Bus Terminal", "Civil Lines", "Airport Road", "Southgate",
    "Hillcrest", "Metro Hub", "Garden Estate", "Ring Junction", "Fort View", "Sports Complex",
]
KINDS = {"Central Station": "transit", "Metro Hub": "transit", "Bus Terminal": "transit",
         "City Hospital": "hospital", "University": "education", "Tech Park": "office",
         "Industrial Area": "industrial", "Airport Road": "airport", "Stadium": "leisure",
         "Mall Road": "commercial", "Old Market": "commercial"}
ROAD = {  # type: (speed km/h, capacity veh/h, cost units per km)
    "street": (40, 900, 12),
    "arterial": (60, 1800, 25),
    "expressway": (80, 3200, 60),
}


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    nodes = []
    grid = {}
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            nid = f"N{i + 1:02d}"
            x = c * SPACING + rng.uniform(-0.25, 0.25)
            y = (ROWS - 1 - r) * SPACING + rng.uniform(-0.25, 0.25)
            name = NAMES[i]
            nodes.append((nid, name, round(x, 3), round(y, 3), KINDS.get(name, "junction")))
            grid[(r, c)] = nid
    # two outskirts villages connected by a single road each (dead ends)
    nodes.append(("N31", "Lakeside Village", round(-1.2, 3), round(2.6, 3), "junction"))
    nodes.append(("N32", "Quarry Hamlet", round((COLS - 1) * SPACING + 1.3, 3), 0.4, "junction"))
    xy = {n[0]: (n[2], n[3]) for n in nodes}

    pairs = []
    for r in range(ROWS):
        for c in range(COLS):
            if c + 1 < COLS:
                pairs.append((grid[(r, c)], grid[(r, c + 1)], "h", r, c))
            if r + 1 < ROWS:
                pairs.append((grid[(r, c)], grid[(r + 1, c)], "v", r, c))
    kept = []
    for a, b, d, r, c in pairs:
        if d == "v" and r == 2 and c not in (1, 4):      # a river: only two crossings
            continue
        if rng.random() < 0.12 and d == "h" and r not in (0, 2):  # a few missing roads
            continue
        kept.append((a, b))
    kept += [("N31", grid[(2, 0)]), ("N32", grid[(4, 5)])]
    # arterials: middle row and column; expressways: two diagonals + ring pieces
    art = {grid[(r, 2)] for r in range(ROWS)} | {grid[(2, c)] for c in range(COLS)}
    expressways = [(grid[(0, 0)], grid[(1, 1)]), (grid[(1, 1)], grid[(3, 3)]),
                   (grid[(3, 3)], grid[(4, 5)]), (grid[(0, 5)], grid[(1, 4)]),
                   (grid[(1, 4)], grid[(3, 1)])]
    rows = []
    eid = 1
    seen = set()
    for a, b in kept + expressways:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        if (a, b) in expressways or (b, a) in expressways:
            rtype = "expressway"
        elif a in art and b in art:
            rtype = "arterial"
        else:
            rtype = "street"
        speed, cap, cost = ROAD[rtype]
        straight = math.dist(xy[a], xy[b])
        dist = round(straight * (1 + rng.uniform(0.03, 0.15)), 3)
        # busier near the middle of the city
        mx = (xy[a][0] + xy[b][0]) / 2; my = (xy[a][1] + xy[b][1]) / 2
        centre = math.dist((mx, my), ((COLS - 1) * SPACING / 2, (ROWS - 1) * SPACING / 2))
        load = max(0.2, min(0.85, 0.75 - 0.08 * centre + rng.uniform(-0.1, 0.1)))
        rows.append((eid, a, b, dist, speed, cap, round(cap * load), rtype, cost))
        eid += 1
    with open(os.path.join(OUT, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "x_km", "y_km", "kind"])
        w.writerows(nodes)
    with open(os.path.join(OUT, "roads.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "from", "to", "distance_km", "speed_limit_kmh", "capacity_vph",
                    "base_volume_vph", "road_type", "build_cost_per_km"])
        w.writerows(rows)
    print(f"wrote {len(nodes)} junctions and {len(rows)} roads to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
