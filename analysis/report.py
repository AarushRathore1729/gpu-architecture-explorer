"""Generate dependency-free HTML/CSV and result-grounded résumé candidates."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from analysis.schema import validate_result
from models.performance_predictor import annotate
from models.roofline import calibration_from_rows


def prepare(data: dict) -> dict:
    validate_result(data)
    data["calibration"] = calibration_from_rows(data["benchmarks"])
    data["prediction_summary"] = annotate(data["benchmarks"], data["calibration"])
    return data


def to_html(data: dict | None = None) -> str:
    if data is not None:
        prepare(data)
    template = (Path(__file__).resolve().parents[1] / "dashboard" / "viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace("__DATA_JSON__", payload)


CSV_FIELDS = ["case_id", "status", "role", "category", "operation", "backend", "dtype", "compute_mode",
              "m", "n", "k", "size_mib", "stride", "block_size", "footprint_bytes", "flops", "logical_bytes",
              "arithmetic_intensity", "latency_ms", "effective_gbps", "tflops", "p10_batch_ms", "p90_batch_ms",
              "coefficient_of_variation", "timing_method", "predicted_ms", "absolute_percentage_error", "error"]


def to_csv(data: dict) -> str:
    validate_result(data)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in data["benchmarks"]:
        timing = row.get("timing", {})
        writer.writerow({**row, **timing, **row.get("prediction", {}), "timing_method": timing.get("method", "")})
    return output.getvalue()


def resume_candidates(data: dict) -> str:
    ok = [r for r in data["benchmarks"] if r["status"] == "ok"]
    name = data["device"]["name"]
    categories = sorted({r["category"] for r in ok if r["category"] != "launch"})
    first = f"- Built a reproducible {name} characterization suite covering {', '.join(categories)}; correctness-checked {len(ok)} measured configurations using CUDA-event timing."
    compute = [r for r in ok if r["category"] == "gemm"]
    copies = [r for r in ok if r["operation"] == "copy" and r["role"] == "calibration"]
    bullets = [first]
    if compute and copies:
        best = max(compute, key=lambda r: r["tflops"])
        bandwidth = max(r["effective_gbps"] for r in copies)
        bullets.append(f"- Measured up to {best['tflops']:.1f} dense {best['compute_mode']} TFLOP/s and {bandwidth:.0f} effective streaming-copy GB/s; visualized arithmetic intensity and precision-dependent empirical rooflines.")
    summary = data.get("prediction_summary", {})
    if summary.get("count"):
        bullets.append(f"- Implemented calibrated compute/memory runtime estimates and evaluated {summary['count']} held-out configurations; exported raw timing samples, model errors, CSV data, and a standalone interactive report.")
    details = f"\n\nRun status: {data.get('status', 'unknown')}\nTimestamp: {data.get('timestamp_utc')}\nSource SHA256: {data.get('source_sha256')}\n"
    if summary.get("mape_percent") is not None:
        details += f"Model MAPE: {summary['mape_percent']:.1f}%; median APE: {summary['median_ape_percent']:.1f}%. Do not describe this as an accurate predictor without reviewing errors.\n"
    details += "\nReview before use: these are measured-suite claims, not prior project history. Custom kernels, traces, and optional modes may be absent if skipped. Bandwidth is logical-byte throughput, not measured DRAM traffic.\n"
    return "# Résumé candidates generated from this run\n\n" + "\n".join(bullets) + details


def write_outputs(data: dict, directory: str | Path):
    prepare(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.html").write_text(to_html(data), encoding="utf-8")
    (directory / "measurements.csv").write_text(to_csv(data), encoding="utf-8")
    (directory / "resume_candidates.md").write_text(resume_candidates(data), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps({"device": data["device"], "calibration": data["calibration"],
                                                        "prediction_summary": data["prediction_summary"]}, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Measurement JSON. Omit for an empty interactive viewer.")
    parser.add_argument("--output", default="results/report.html")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text()) if args.input else None
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_html(data), encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    main()
