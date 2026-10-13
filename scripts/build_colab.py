"""Build a self-contained Colab notebook using only the Python standard library."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def code(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or any(p in {"__pycache__", ".pytest_cache", ".git", ".venv"} for p in path.parts):
                continue
            relative = path.relative_to(ROOT)
            if path.suffix in {".ipynb", ".zip", ".pyc", ".tmp"} or any(p.endswith(".egg-info") for p in path.parts):
                continue
            if relative.parts[0] == "results" and relative.name not in {"README.md", "explorer.html", "local_cpu_tests.xml"}:
                continue
            archive.write(path, (Path(ROOT.name) / relative).as_posix())
    payload = base64.b64encode(buffer.getvalue()).decode()
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    setup = '''import base64, hashlib, importlib.util, io, os, subprocess, sys, zipfile
from pathlib import Path
from importlib import metadata
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > A100 GPU, then rerun.")
print("GPU:", torch.cuda.get_device_name(0), "| PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
SOURCE_B64 = "PAYLOAD"
SOURCE_SHA256 = "DIGEST"
source_bytes = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256, "Embedded source checksum failed"
WORKDIR = Path("/content/gpu-architecture-explorer")
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    for name in archive.namelist():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or p.parts[0] != "gpu-architecture-explorer":
            raise RuntimeError("Unsafe embedded source path")
    archive.extractall("/content")
os.chdir(WORKDIR)
if importlib.util.find_spec("pytest") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "pytest>=8,<10"], check=True)
if importlib.util.find_spec("triton") is None:
    candidates = [r.split(";")[0].strip() for r in (metadata.requires("torch") or []) if r.lower().startswith("triton==")]
    if not candidates:
        raise RuntimeError("Matching Triton is missing. Use a standard Colab CUDA PyTorch runtime.")
    subprocess.run([sys.executable, "-m", "pip", "install", candidates[0]], check=True)
print("Embedded repository ready:", WORKDIR)
'''.replace("PAYLOAD", payload).replace("DIGEST", digest)
    cells = [
        markdown("# GPU Architecture Explorer — A100 run\n\nSelect **Runtime → Change runtime type → A100 GPU**, then **Run all**. This notebook embeds the complete repository. It tests correctness before benchmarking, writes raw CUDA-event measurements, and downloads the full source/results ZIP.\n\nKeep `PRESET = \"full\"` for the complete suite. Results are real only after this notebook runs successfully; the supplied viewer contains no sample GPU numbers.\n"),
        code(setup),
        markdown("## Configure and run correctness tests\n\nThe full preset attempts 55 cases on an A100 with all four precision modes. CUDA graph timing amortizes Python submission overhead; any eager fallback is recorded. Test failures skip benchmarking and are still exported for diagnosis.\n"),
        code('''from datetime import datetime, timezone
PRESET = "full"           # "quick" is a smaller preflight suite
TIMING = "cuda_graph"     # use "eager" if graph capture is unreliable
PROFILE = True            # optional separate PyTorch CPU/CUDA trace
RUN_TAG = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = WORKDIR / "results" / RUN_TAG
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_logged(command, log_path):
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=WORKDIR, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()

TEST_EXIT = run_logged([sys.executable, "-m", "pytest", "-q",
                       "--junitxml=" + str(RESULTS_DIR / "gpu_test_results.xml")],
                      RESULTS_DIR / "tests.log")
print("Correctness test exit code:", TEST_EXIT)
'''),
        markdown("## Measure GPU workloads\n\nCompilation, allocations, and correctness checks are outside timed regions. The model is calibrated on separate sizes; strided access is excluded from prediction-error scoring. Checkpoints preserve successful cases if a later case fails.\n"),
        code('''if TEST_EXIT == 0:
    command = [sys.executable, "-m", "benchmarks.run", "--preset", PRESET,
               "--timing", TIMING, "--output", str(RESULTS_DIR / "run.json")]
    if PROFILE:
        command.append("--profile")
    BENCH_EXIT = run_logged(command, RESULTS_DIR / "benchmark.log")
    print("Benchmark exit code:", BENCH_EXIT)
else:
    BENCH_EXIT = None
    print("Benchmarks skipped because correctness tests failed. Export the diagnostic bundle below.")
'''),
        markdown("## Inspect and export\n\nThe exported ZIP includes the source repository and measured files. Open its `report.html` in a browser. Send back the ZIP for verification and final résumé bullets. If the run failed, send the same diagnostic ZIP.\n"),
        code('''import html, json
from IPython.display import HTML, display
from google.colab import files

measurement = RESULTS_DIR / "run.json"
if measurement.exists():
    data = json.loads(measurement.read_text())
    successful = [r for r in data.get("benchmarks", []) if r.get("status") == "ok"]
    failed = [r for r in data.get("benchmarks", []) if r.get("status") == "failed"]
    print("Run:", data.get("status"), "| validated measurements:", len(successful), "| failed:", len(failed))
    print("Calibration:", json.dumps(data.get("calibration", {}), indent=2))
    print("Held-out model errors:", json.dumps(data.get("prediction_summary", {}), indent=2))
    report = RESULTS_DIR / "report.html"
    if report.exists():
        display(HTML('<iframe title="GPU Architecture Explorer" style="width:100%;height:900px;border:1px solid #ccd8e7;border-radius:10px" srcdoc="' + html.escape(report.read_text(), quote=True) + '"></iframe>'))
else:
    print("No measurements available. The archive contains logs for diagnosis.")

(RESULTS_DIR / "notebook_status.json").write_text(json.dumps({"test_exit_code": TEST_EXIT,
    "benchmark_exit_code": BENCH_EXIT, "preset": PRESET, "timing": TIMING, "run_tag": RUN_TAG}, indent=2))
archive_path = Path("/content/gpu_architecture_explorer_results.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(WORKDIR.rglob("*")):
        if not path.is_file() or any(p in {"__pycache__", ".pytest_cache", ".git", ".venv"} for p in path.parts):
            continue
        if path.suffix in {".pyc", ".tmp", ".zip"} or any(p.endswith(".egg-info") for p in path.parts):
            continue
        archive.write(path, (Path(WORKDIR.name) / path.relative_to(WORKDIR)).as_posix())
print("Download:", archive_path.name)
files.download(str(archive_path))
'''),
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "colab": {"name": "GPU_Architecture_Explorer_Colab.ipynb"}, "accelerator": "GPU"},
                "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(cells):
        cell["id"] = f"explorer-{index:02d}"
    target = ROOT / "notebooks" / "GPU_Architecture_Explorer_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Built {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
