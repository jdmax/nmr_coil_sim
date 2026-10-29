"""Precompute the preset coils at full resolution.

Writes
  web/data/default.json   preset data loaded by the GitHub Pages front end
  explorer.html           the same explorer as one self-contained file (no Pyodide;
                          handy for sharing a fixed set of results)

The coil definitions live in explorer_data.presets(); add or change coils there.

Run: python build_explorer.py      (about a minute)
"""
import json
import os
import time
import explorer_data as ed

CELLS = [(15.0, 50.0), (20.0, 50.0)]


def main():
    t0 = time.time()
    data = dict(cells=[ed.build_cell(d, L, "full") for d, L in CELLS],
                lut=ed.lut(), freqs=dict(p=ed.F_P / 1e6, d=ed.F_D / 1e6),
                types={k: dict(label=v["label"], params=[dict(key=p[0], label=p[1], unit=p[2], default=p[3])
                                                          for p in v["params"]])
                       for k, v in ed.TYPES.items()})
    blob = json.dumps(data, separators=(",", ":"))
    os.makedirs("web/data", exist_ok=True)
    with open("web/data/default.json", "w", encoding="utf-8") as fh:
        fh.write(blob)
    html = open("explorer_template.html", encoding="utf-8").read()
    with open("explorer.html", "w", encoding="utf-8") as fh:
        fh.write(html.replace("/*__DATA__*/null", blob))
    print(f"wrote web/data/default.json and explorer.html ({len(blob)/1e6:.2f} MB) "
          f"in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
