"""Launch the Streamlit app headless, check it answers HTTP 200, run an optimisation
in a real browser (Playwright + headless Chromium), save screenshots to
docs/screenshots/ and stop the server.

    pip install -r requirements-dev.txt
    python -m playwright install chromium
    python scripts/screenshot_app.py [--port 8531]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"


def wait_http(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return r.status
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8531)
    a = ap.parse_args()
    url = f"http://localhost:{a.port}"
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(a.port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        status = wait_http(url)
        print(f"GET {url} -> HTTP {status}")
        health = wait_http(url + "/_stcore/health")
        print(f"GET {url}/_stcore/health -> HTTP {health}")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1250}, device_scale_factor=1)
            page.goto(url, wait_until="networkidle")
            page.get_by_text("Run optimisation").first.wait_for(timeout=60_000)
            time.sleep(1.5)
            page.screenshot(path=str(OUT / "app_start.png"))
            page.get_by_test_id("stSelectbox").first.click()
            page.get_by_role("option", name="C-clamp (PLA, N/mm/MPa)").click()
            time.sleep(2.5)
            page.get_by_role("button", name="Run optimisation").click()
            page.get_by_text("Downloads").wait_for(timeout=300_000)
            time.sleep(3)
            page.screenshot(path=str(OUT / "app_design.png"))
            for tab, name in [("Stress", "app_stress.png"), ("3D / STL", "app_stl.png"),
                              ("Convergence", "app_convergence.png")]:
                page.get_by_role("tab", name=tab).click()
                time.sleep(2.5)
                page.screenshot(path=str(OUT / name))
            errors = page.locator("[data-testid='stException']").count()
            print(f"exceptions shown in app: {errors}")
            browser.close()
        print(f"screenshots written to {OUT}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("streamlit server stopped")


if __name__ == "__main__":
    main()
