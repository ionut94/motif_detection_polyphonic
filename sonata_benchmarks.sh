#!/bin/bash

# Benchmark script to run sonata src/sonata_results.py with delta and gamma combos

RESULTS_DIR="results"
SCRIPT="src/sonata_results.py"

# Extract delta and gamma values from results folder
for delta in 0 1 2 3 4 8 16 24; do
    for gamma in 0 1 2 3 4 8 16 24; do
        echo "Running benchmark with delta=$delta, gamma=$gamma"
        /Users/imoraru/Work/music_paper/.venv/bin/python "$SCRIPT" --delta "$delta" --gamma "$gamma"
    done
done

echo "All benchmarks completed"