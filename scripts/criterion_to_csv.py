"""Append Criterion's timing medians to results/results.csv as time_per_step rows.

Usage: python scripts/criterion_to_csv.py "commit,cpu,os,toolchain,date"

Reads target/criterion/<group>/<library>/new/estimates.json for every group named
<Scenario>_<Filter>_<precision> (from benches/compare.rs). The step count per iteration comes
from tests/vectors/<Scenario>/model.json, and adskalman's version from Cargo.lock.
"""
import csv
import json
import re
import sys
import tomllib
from pathlib import Path

GROUP = re.compile(r"^(S\d)_([A-Z]+)_(float32|float64)$")

env = sys.argv[1].split(",")
if len(env) != 5:
    sys.exit('expected "commit,cpu,os,toolchain,date"')
lock = tomllib.load(open("Cargo.lock", "rb"))
adskalman_version = next(p["version"] for p in lock["package"] if p["name"] == "adskalman")

rows = 0
Path("results").mkdir(exist_ok=True)
with open("results/results.csv", "a", newline="") as f:
    out = csv.writer(f)
    for path in sorted(Path("target/criterion").glob("*/*/new/estimates.json")):
        group, library = path.parts[-4], path.parts[-3]
        match = GROUP.match(group)
        if not match:
            continue  # other benchmarks, such as benches/filters.rs
        scenario, filt, precision = match.groups()
        model = json.load(open(f"tests/vectors/{scenario}/model.json"))
        steps = model["runs"] * model["steps"]
        ns_per_step = json.load(open(path))["median"]["point_estimate"] / steps
        version = env[0] if library.startswith("typed-kalman") else adskalman_version
        out.writerow([library, version, scenario, filt, precision,
                      "time_per_step", f"{ns_per_step:.2f}", "ns", *env])
        rows += 1
print(f"appended {rows} time_per_step rows")
