#!/usr/bin/env bash
# Builds the firmware images in bench/firmware and prints flash_bytes and ram_bytes rows
# (results/results.csv schema) for each library.
#
# Usage: scripts/firmware_sizes.sh "commit,cpu,os,toolchain,date"
#
# flash_bytes = .vector_table + .text + .rodata + .data (everything stored in flash; the
# guide's formula plus the vector table, which is identical in both images).
# ram_bytes = .data + .bss: statically allocated RAM only. The filter's state lives on the
# stack, which this doesn't measure.
set -euo pipefail
cd "$(dirname "$0")/../bench/firmware"
ENV="$1"
COMMIT="${ENV%%,*}"

cargo build --release --quiet
SIZE="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-size"
ADSKALMAN_VERSION="$(grep -A1 '^name = "adskalman"$' Cargo.lock | sed -n 's/^version = "\(.*\)"$/\1/p')"

section() { # <elf> <section name>: prints the section's size, or 0 if it's absent
  "$SIZE" -A "$1" | awk -v name="$2" '$1 == name { print $2; found = 1 } END { if (!found) print 0 }'
}

for pair in "fw_kalman_rs:kalman-rs:$COMMIT" "fw_adskalman:adskalman-joseph:$ADSKALMAN_VERSION"; do
  IFS=: read -r bin library version <<< "$pair"
  elf="target/thumbv7em-none-eabihf/release/$bin"
  vectors=$(section "$elf" .vector_table)
  text=$(section "$elf" .text)
  rodata=$(section "$elf" .rodata)
  data=$(section "$elf" .data)
  bss=$(section "$elf" .bss)
  # An image without code means the linker script was missing and everything was discarded.
  if [ "$vectors" -eq 0 ] || [ "$text" -eq 0 ]; then
    echo "error: $bin has no vector table or code; was it linked without link.x?" >&2
    exit 1
  fi
  echo "$library,$version,S2,KF,float32,flash_bytes,$((vectors + text + rodata + data)),bytes,$ENV"
  echo "$library,$version,S2,KF,float32,ram_bytes,$((data + bss)),bytes,$ENV"
done
