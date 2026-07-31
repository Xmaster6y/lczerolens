#!/usr/bin/env bash
set -euo pipefail

# Google Drive identifiers and checksums are versioned in test-fixtures.sha256.
# This setup command is the only test helper that accesses the network.
sources=(
  "tinygyal-8.pb.gz:1Ssl4JanqzQn3p-RoHRDk_aApykl-SukE"
  "maia-1100.pb.gz:1erxB3tULDURjpPhiPWVGr6X986Q8uE6U"
  "t1-smolgen-512x15x8h-distilled-swa-3395000.pb.gz:1YqqANK-wuZIOmMweuK_oCU7vfPN7G_Z6"
  "test_stockfish_10.jsonl:15-eGN7Hz2NM6aEMRaQrbW3ScxxQpAqa5"
)

for source in "${sources[@]}"; do
  fixture="${source%%:*}"
  source_id="${source#*:}"
  if [[ ! -f "assets/$fixture" ]]; then
    uv run gdown "$source_id" -O "assets/$fixture"
  fi
done

(cd assets && shasum -a 256 -c test-fixtures.sha256)
