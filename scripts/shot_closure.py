"""Screenshot of the closure scenario: close two roads on the best route, add an incident."""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
OUT = "docs/screenshots"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1500, "height": 2300})
    page.goto(URL)
    page.wait_for_selector("text=Smart City Traffic", timeout=30000)
    page.wait_for_timeout(3000)
    box = page.get_by_label("Close roads")
    for opt in ("#48 Tech Park - Civil Lines", "#36 Civil Lines - Airport Road"):
        box.click()
        page.keyboard.type(opt[:6])
        page.get_by_role("option", name=opt).first.click()
        page.wait_for_timeout(1500)
    page.keyboard.press("Escape")
    page.wait_for_timeout(2500)
    page.screenshot(path=f"{OUT}/07_closure_commuter.png")
    page.get_by_role("tab", name="🏛 City administrator").click()
    page.wait_for_timeout(3000)
    page.screenshot(path=f"{OUT}/08_closure_admin.png")
    b.close()
print("done")
