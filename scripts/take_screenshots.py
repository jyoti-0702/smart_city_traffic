"""Capture screenshots of the running app for the report (needs playwright + chrome)."""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = "docs/screenshots"


def settle(page, ms=2500):
    page.wait_for_timeout(ms)


def pick(page, label, option):
    page.get_by_label(label).first.click()
    page.get_by_role("option", name=option).first.click()
    settle(page)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 2300})
    page.goto(URL)
    page.wait_for_selector("text=Smart City Traffic", timeout=30000)
    settle(page, 4000)
    page.screenshot(path=f"{OUT}/01_commuter_route.png")
    # admin tab: traffic
    page.get_by_role("tab", name="🏛 City administrator").click(); settle(page, 3000)
    page.screenshot(path=f"{OUT}/02_admin_traffic.png")
    page.get_by_text("Critical roads (bridges)").click(); settle(page, 2500)
    page.screenshot(path=f"{OUT}/03_admin_critical_roads.png")
    page.get_by_text("Minimum spanning tree", exact=True).first.click(); settle(page, 2500)
    page.screenshot(path=f"{OUT}/04_admin_mst.png")
    page.get_by_role("tab", name="🔍 Network explorer").click(); settle(page, 3000)
    page.screenshot(path=f"{OUT}/05_explorer_bfs.png")
    page.get_by_role("tab", name="⏱ Performance analysis").click(); settle(page, 3000)
    page.screenshot(path=f"{OUT}/06_performance.png")
    b.close()
print("done")
