#!/bin/bash

# Benchmark script to run sonata src/sonata_results.py with delta and gamma combos

RESULTS_DIR="results"
SCRIPT="src/sonata_results.py"

# Extract delta and gamma values from results folder
# Delta: 0 to 6 (since 6 is max difference in mod 12)
for delta in 0 1 2 3 4 5 6; do
    # Gamma: 0 to 24 (reasonable spread)
    for gamma in 0 2 4 8 16 24; do
        echo "Running benchmark with delta=$delta, gamma=$gamma"
        /Users/imoraru/Work/music_paper/.venv/bin/python "$SCRIPT" --delta "$delta" --gamma "$gamma" --runs 5
    done
done

echo "All benchmarks completed"