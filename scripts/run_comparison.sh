#!/usr/bin/env bash
# Produces every comparison number and the README table in one go.
#
# Run it on the dedicated benchmark machine (fixed CPU frequency, nothing else running) for
# published numbers; anywhere else, treat the timings as indicative only.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV="$(git rev-parse --short HEAD),$(uname -m),$(uname -s),$(rustc --version | cut -d' ' -f2),$(date +%F)"
mkdir -p results
if [ ! -f results/results.csv ]; then
  echo "library,library_version,scenario,filter,precision,metric,value,unit,commit,cpu,os,toolchain,date" \
    > results/results.csv
fi

cargo bench --bench compare
python3 scripts/criterion_to_csv.py "$ENV"
cargo run --release --example accuracy -- "$ENV" >> results/results.csv
scripts/firmware_sizes.sh "$ENV" >> results/results.csv
python3 scripts/make_table.py results/results.csv typed-kalman
