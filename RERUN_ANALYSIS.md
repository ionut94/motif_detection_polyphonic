# Benchmark Rerun Analysis (Delta Semantics Fix)

## Overview

The full benchmark suite was rerun with the corrected `delta` semantics.

- **Old Semantics**: `delta` = Maximum number of mismatching positions.
- **New Semantics**: `delta` = Maximum allowed pitch difference (semitones) for any single aligned note pair.

## Comparison Results

### Case 1: High Delta, High Gamma (delta=8, gamma=24)

- **Old Occurrences**: 2,356,658
- **New Occurrences**: 2,393,821
- **Difference**: +37,163 (+1.6%)
- **Interpretation**: The new logic allows matches that have many small deviations (e.g., 9 notes off by 1 semitone). The old logic rejected these because the *count* of mismatches (9) exceeded delta (8). The new logic accepts them because the *max difference* (1) is within delta (8) and the total error (9) is within gamma (24).

### Case 2: Low Delta, High Gamma (delta=2, gamma=24)

- **Old Occurrences**: 163,721
- **New Occurrences**: 122,054
- **Difference**: -41,667 (-25.4%)
- **Interpretation**: The new logic rejects matches that have a few large deviations. For example, a match with 1 note off by 5 semitones. The old logic accepted this because the *count* of mismatches (1) was within delta (2). The new logic rejects it because the *max difference* (5) exceeds delta (2).

## Conclusion

The fix has been successfully applied and verified. The algorithm now correctly implements the "per-note tolerance" semantics for `delta`, behaving more strictly against large single-note deviations and more permissively towards cumulative small deviations (bounded by `gamma`).
