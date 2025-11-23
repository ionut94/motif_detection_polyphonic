#!/bin/bash

# Benchmark script to run sonata src/sonata_results.py with delta and gamma combos

RESULTS_DIR="results"
SCRIPT="src/sonata_results.py"

# Extract delta and gamma values from results folder
# Delta: 0 to 6 (since 6 is max difference in mod 12)
for delta in 2; do
    # Gamma: 0 to 24 (reasonable spread)
    for gamma in 8; do
        echo "Running benchmark with delta=$delta, gamma=$gamma"
        /Users/imoraru/Work/music_paper/.venv/bin/python "$SCRIPT" --delta "$delta" --gamma "$gamma" --runs 5
    done
done

echo "All benchmarks completed"