"""Usage: python scripts/make_table.py results/results.csv <our-library-name>"""
import csv
import sys
from collections import defaultdict

LOWER_IS_BETTER = {"time_per_step", "cycles_per_step", "rmse", "heap_allocations",
                   "peak_memory", "flash_bytes", "ram_bytes"}
HIGHER_IS_BETTER = {"steps_to_failure"}

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
ours = sys.argv[2]
libs = [ours] + sorted({r["library"] for r in rows} - {ours})

cells = defaultdict(dict)
for r in rows:
    key = (r["scenario"], r["filter"], r["precision"], r["metric"], r["unit"])
    cells[key][r["library"]] = float(r["value"])


def ratio(metric, vals):
    # Libraries named "<ours>-something" are our own variants (e.g. another backend)
    others = [v for lib, v in vals.items() if not lib.startswith(ours)]
    if ours not in vals or not others:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        best = min(others)
        return "–" if vals[ours] == 0 else f"{best / vals[ours]:.2f}x"
    if metric in HIGHER_IS_BETTER:
        best = max(others)
        return "–" if best == 0 else f"{vals[ours] / best:.2f}x"
    return "–"  # nees, max_abs_diff: read the values directly


print("| Scenario | Filter | Precision | Metric | " + " | ".join(libs) + " | Ours vs best other |")
print("|" + "---|" * (len(libs) + 5))
for (scen, filt, prec, metric, unit), vals in sorted(cells.items()):
    values = [f"{vals[lib]:.4g}" if lib in vals else "n/a" for lib in libs]
    print(f"| {scen} | {filt} | {prec} | {metric} ({unit}) | " + " | ".join(values)
          + f" | {ratio(metric, vals)} |")
