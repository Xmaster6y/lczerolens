#!/usr/bin/env bash
set -euo pipefail

# Google Drive identifiers and checksums are versioned in test-fixtures.sha256.
# This setup command is the only test helper that accesses the network.
sources=(
  "tinygyal-8.pb.gz:1Ssl4JanqzQn3p-RoHRDk_aApykl-SukE"
  "384x30-2022_0108_1903_17_608.pb.gz:1WzBQV_zn5NnfsG0K8kOion0pvWxXhgKM"
  "maia-1100.pb.gz:1erxB3tULDURjpPhiPWVGr6X986Q8uE6U"
  "t1-smolgen-512x15x8h-distilled-swa-3395000.pb.gz:1YqqANK-wuZIOmMweuK_oCU7vfPN7G_Z6"
  "test_stockfish_10.jsonl:15-eGN7Hz2NM6aEMRaQrbW3ScxxQpAqa5"
)

for source in "${sources[@]}"; do
  fixture="${source%%:*}"
  source_id="${source#*:}"
  if [[ -f "assets/$fixture" ]] && ! (cd assets && grep -F "  $fixture" test-fixtures.sha256 | shasum -a 256 -c - >/dev/null); then
    rm "assets/$fixture"
  fi
  if [[ ! -f "assets/$fixture" ]]; then
    uv run --group dev gdown "$source_id" -O "assets/$fixture"
  fi
done

(cd assets && shasum -a 256 -c test-fixtures.sha256)
