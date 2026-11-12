"""Assemble the GitHub Pages site into _site/ and (optionally) serve it locally.

The site is web/ plus the Python modules the in-browser worker imports, copied to
_site/py/ so the browser runs exactly the code in this repository.

    python serve_web.py            build _site/ and serve it at http://localhost:8000
    python serve_web.py --build    build only (what the GitHub Action runs)
    python serve_web.py --port 8080
"""
import argparse
import functools
import hashlib
import http.server
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "_site"
PY_MODULES = ["coilsim.py", "explorer_data.py"]


def build() -> Path:
    if SITE.exists():
        shutil.rmtree(SITE)
    shutil.copytree(ROOT / "web", SITE)
    (SITE / "py").mkdir()
    for name in PY_MODULES:
        shutil.copy2(ROOT / name, SITE / "py" / name)
    # Cache-bust: GitHub Pages serves with max-age=600, so a browser can pair a new index.html with a stale
    # app.js or style.css. Tag their URLs with a content hash so any change gets a fresh URL.
    page = SITE / "index.html"
    html = page.read_text(encoding="utf-8")
    for name in ("app.js", "style.css"):
        tag = hashlib.sha256((SITE / name).read_bytes()).hexdigest()[:10]
        html = html.replace(f'"{name}"', f'"{name}?v={tag}"')
    page.write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").touch()
    if not (SITE / "data" / "default.json").exists():
        raise SystemExit("web/data/default.json is missing: run python build_explorer.py first")
    print(f"built {SITE}")
    return SITE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build _site/ and exit")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    site = build()
    if args.build:
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving http://localhost:{args.port}  (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
