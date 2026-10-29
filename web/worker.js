// Runs the repository's Python (coilsim.py + explorer_data.py) in Pyodide, off the main thread.
// The site build copies those files into ./py/ next to this worker.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const PY_FILES = ["coilsim.py", "explorer_data.py"];

let ready = null;

async function boot() {
  postMessage({ type: "progress", text: "Loading Pyodide…" });
  importScripts(PYODIDE + "pyodide.js");
  const py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "progress", text: "Loading NumPy…" });
  await py.loadPackage("numpy");
  postMessage({ type: "progress", text: "Loading coil simulation code…" });
  for (const f of PY_FILES) {
    const r = await fetch("py/" + f);
    if (!r.ok) throw new Error(`could not fetch py/${f} (${r.status})`);
    py.FS.writeFile(f, await r.text());
  }
  py.runPython("import sys; sys.path.insert(0, '.')\nimport numpy, explorer_data");
  const numpyVersion = py.runPython("numpy.__version__");
  postMessage({ type: "ready", version: py.version, numpy: numpyVersion });
  return py;
}

ready = boot().catch(err => { postMessage({ type: "error", error: String(err && err.message || err) }); throw err; });

onmessage = async e => {
  const msg = e.data;
  if (msg.type !== "compute") return;
  try {
    const py = await ready;
    const fn = py.globals.get("explorer_data").compute_entry_json;
    const json = fn(JSON.stringify(msg.spec), JSON.stringify(msg.cell), "web");
    fn.destroy();
    postMessage({ type: "result", id: msg.id, json });
  } catch (err) {
    postMessage({ type: "error", id: msg.id, error: String(err && err.message || err) });
  }
};
