"""Precompute the preset coils at full resolution.

Writes
  web/data/default.json   preset data loaded by the GitHub Pages front end (index.html)
  web/explorer.html       the Coil Raster Explorer as one self-contained page: the
                          preset coils only, no Pyodide, works offline and from any
                          static host (GitHub Pages serves it at /explorer.html)

The coil definitions live in explorer_data.presets(); add or change coils there.

    python build_explorer.py              recompute the presets (about a minute), write both
    python build_explorer.py --from-json  rebuild web/explorer.html from the existing
                                          web/data/default.json (seconds)
"""
import argparse
import json
import os
import time

CELLS = [(15.0, 50.0), (20.0, 50.0)]
DATA_PATH = "web/data/default.json"
EXPLORER_PATH = "web/explorer.html"

# explorer_template.html is a page body (its own <title>, <style>, markup and script);
# this wraps it in a complete document for static hosting.
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Coil Raster Explorer: raster sampling, deuteron signal bias and tunability of cw-NMR coils on polarized target cells (preset coils, single file).">
<meta name="author" content="J. Maxwell">
<style>
:root { padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
html, body { margin: 0; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
</head>
<body>
"""
FOOT_LINKS = """  <footer class="notes" style="font-size:0.8rem">
    Single-file version with the preset coils. To design and compute your own coils in the browser, use the
    <a href="./" style="color:var(--copper)">full explorer</a>. Source and tests:
    <a href="https://github.com/jdmax/nmr_coil_sim" style="color:var(--copper)">github.com/jdmax/nmr_coil_sim</a>.<br>
    By <a href="https://orcid.org/0000-0003-2710-4646" style="color:var(--copper)">J. Maxwell</a>. © 2026 J. Maxwell,
    <a href="https://github.com/jdmax/nmr_coil_sim/blob/main/LICENSE" style="color:var(--copper)">MIT license</a>.
  </footer>
</div>

<script>"""
TAIL = """
</body>
</html>
"""


def compute_data() -> str:
    import explorer_data as ed
    data = dict(cells=[ed.build_cell(d, L, "full") for d, L in CELLS],
                lut=ed.lut(), freqs=dict(p=ed.F_P / 1e6, d=ed.F_D / 1e6),
                types={k: dict(label=v["label"], params=[dict(key=p[0], label=p[1], unit=p[2], default=p[3])
                                                          for p in v["params"]])
                       for k, v in ed.TYPES.items()})
    blob = json.dumps(data, separators=(",", ":"))
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        fh.write(blob)
    return blob


def write_explorer(blob: str) -> None:
    body = open("explorer_template.html", encoding="utf-8").read()
    marker = "</div>\n\n<script>"
    if body.count(marker) != 1:
        raise SystemExit("explorer_template.html: could not find the end of the page markup")
    body = body.replace(marker, FOOT_LINKS)
    html = HEAD + body.replace("/*__DATA__*/null", blob) + TAIL
    with open(EXPLORER_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", action="store_true",
                    help=f"reuse {DATA_PATH} instead of recomputing the presets")
    args = ap.parse_args()
    t0 = time.time()
    if args.from_json:
        blob = open(DATA_PATH, encoding="utf-8").read()
    else:
        blob = compute_data()
    write_explorer(blob)
    print(f"wrote {'' if args.from_json else DATA_PATH + ' and '}{EXPLORER_PATH} "
          f"({len(blob)/1e6:.2f} MB of data) in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
