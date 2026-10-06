"""Fail if any of our benchmarks got slower than a baseline by more than a threshold, with
95% confidence.

Usage: python scripts/check_regression.py 0.10

Run after `cargo bench --bench compare -- --baseline <name>`, which writes each benchmark's
relative change to target/criterion/<group>/<library>/change/estimates.json. Only our own
libraries (kalman-rs, kalman-rs-*) in benches/compare.rs groups are checked; the baselines'
timings are context. Clear target/criterion before the baseline run: change files from earlier
runs stay on disk and would be read as if they were new.
"""
import json
import re
import sys
from pathlib import Path

GROUP = re.compile(r"^S\d_[A-Z]+_float(32|64)$")

threshold = float(sys.argv[1])
regressions = []
checked = 0
for path in sorted(Path("target/criterion").glob("*/*/change/estimates.json")):
    group, library = path.parts[-4], path.parts[-3]
    if not GROUP.match(group) or not library.startswith("kalman-rs"):
        continue
    mean = json.load(open(path))["mean"]
    change, lower = mean["point_estimate"], mean["confidence_interval"]["lower_bound"]
    checked += 1
    # Flag only when even the low end of Criterion's 95% interval is over the threshold.
    # Back-to-back runs of unchanged code vary by up to about 10%, so the point estimate
    # alone would fail pull requests at random.
    regressed = lower > threshold
    flag = "REGRESSION" if regressed else "ok"
    print(f"{group}/{library}: {change:+.1%} (95% CI from {lower:+.1%}) {flag}")
    if regressed:
        regressions.append(f"{group}/{library}")

if checked == 0:
    sys.exit("no benchmark changes found; did the baseline run?")
if regressions:
    sys.exit(f"{len(regressions)} benchmark(s) slower by more than {threshold:.0%}: "
             + ", ".join(regressions))
print(f"no regressions over {threshold:.0%} in {checked} benchmarks")
