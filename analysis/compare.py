"""Compare matching case IDs; do not attribute all differences to architecture."""
import argparse
import json
from pathlib import Path

from analysis.schema import validate_result


def compare(a: dict, b: dict) -> list[dict]:
    validate_result(a)
    validate_result(b)
    right = {r["case_id"]: r for r in b["benchmarks"] if r["status"] == "ok"}
    output = []
    for first in a["benchmarks"]:
        second = right.get(first["case_id"])
        if first["status"] != "ok" or second is None:
            continue
        if first["timing"]["method"] != second["timing"]["method"]:
            continue
        output.append({"case_id": first["case_id"], "a_latency_ms": first["latency_ms"],
                       "b_latency_ms": second["latency_ms"], "a_over_b_latency": first["latency_ms"] / second["latency_ms"]})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args()
    a, b = [json.loads(Path(path).read_text()) for path in [args.a, args.b]]
    print(json.dumps({"a_device": a["device"]["name"], "b_device": b["device"]["name"],
                      "comparison": compare(a, b), "note": "Clocks, GPU sharing, cache behavior, and software versions can differ."}, indent=2))


if __name__ == "__main__":
    main()
