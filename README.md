# Melodic Motif Finder

A Python implementation of an algorithm for finding melodic motifs in MIDI files with bounded mismatches.

## Overview

This project implements an algorithm for efficiently finding melodic motifs in MIDI files, as described in the academic paper. The algorithm allows for bounded mismatches in both the number of different positions (delta) and the Sum of Absolute Differences (gamma), making it robust for musical pattern matching where exact matches might be rare.

## Algorithm

The implementation follows the algorithm described in the paper, which involves the following key steps:

1.  **Preprocessing (T -> T~, M -> P~):**
    *   Extract melodic parts (channels) from the MIDI file `T`.
    *   For each channel, create a `tilde{T}` sequence where each element is either a single pitch class (0-11) for a note or a *set* of unique pitch classes for notes occurring simultaneously (a chord/non-solid symbol).
    *   Preprocess the input motif `M` (list of MIDI pitches) similarly to get its pitch class sequence `P~` (motifs are assumed to be solid, i.e., contain no chords).

2.  **Solid Equivalents (T~ -> T_S, P~ -> P):**
    *   Convert each `tilde{T}` into its solid equivalent string `T_S`. Solid symbols (single pitch classes) are mapped to characters (e.g., 'A'-'L'). Non-solid symbols (sets of pitch classes) are replaced by unique placeholder characters (e.g., `$`<sub>d</sub> from Unicode Private Use Area). A `loc_map` is stored, mapping each `$`<sub>d</sub> back to its original pitch class set.
    *   Convert the motif pitch class sequence `P~` into its solid string `P` using the same character mapping as for solid symbols in `T_S`.

3.  **Combined String (S = T_S #_1 P #_2):**
    *   For each channel's `T_S`, create the combined string `S = T_S #_1 P #_2`, where `#_1` and `#_2` are unique separator characters not present in the alphabet of `T_S` or `P`.

4.  **Matching Table (M):**
    *   Define a matching function `M(char1, char2)` based on the algorithm's rules:
        *   Solid vs. Solid: Match if equal pitch class. Mismatch otherwise, calculate difference `min((pc1-pc2)%12, (pc2-pc1)%12)` for gamma.
        *   Non-solid (`$`<sub>d</sub>) vs. Solid (`p`): Match if `p`'s pitch class is in the set `loc_map[$`<sub>d</sub>]`. Mismatch otherwise, calculate minimum difference between `p` and elements in the set for gamma.
        *   Separator vs. Any: Match only if identical separators.
        *   Other cases (e.g., `$`<sub>d1</sub> vs `$`<sub>d2</sub>): Defined as mismatch.

5.  **Suffix Tree Construction:**
    *   Construct the suffix tree for the combined string `S` using Ukkonen's algorithm (O(|S|) time).

6.  **LCE Queries (LCE_k(S, i, |T_S|+1, M)):**
    *   For each potential start position `i` in `T_S` (from 0 to |T_S| - |P|), perform a Longest Common Extension query `lce_k_gamma_query(i, |T_S|+1, delta, gamma, M)`. This query compares the suffix of `S` starting at `i` (part of `T_S`) with the suffix starting at `|T_S|+1` (which is `P`).
    *   The `lce_k_gamma_query` uses the suffix tree structure and the matching function `M` to find the length of the longest common prefix between the two suffixes, allowing up to `delta` mismatches and a cumulative mismatch difference up to `gamma`.

7.  **Identify Occurrences:**
    *   If the `lce_k_gamma_query` returns a length equal to the length of the motif `P` (|P|), then an occurrence is reported starting at index `i` in the original `T_S` for that channel.

## Requirements

- Python 3.6+
- mido library (for MIDI file processing)
- numpy
- matplotlib (for analysis and visualization)
- psutil (optional, for memory usage tracking)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ionut94/motif_detection_polyphonic.git
   cd motif_detection_polyphonic
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── data/                  # MIDI files and other input data
│   ├── example1chords.mid
│   └── twinkle.mid
├── docs/                  # TODO: Documentation files 
├── results/               # Results from analysis
│   └── plots/             # Generated plots
│       ├── complexity_analysis.png
│       ├── delta_complexity.png
│       └── gamma_complexity.png
├── src/                   # Source code
│   ├── benchmark.py       # Tool for running performance tests
│   ├── complexity_analysis.py    # Time/space complexity analysis
│   ├── main.py            # Entry point for the CLI application
│   ├── midi_processor.py  # MIDI file parsing and conversion
│   ├── motif_finder.py    # Core algorithm implementation
│   ├── param_complexity_analysis.py  # Parameter-based analysis
│   ├── suffix_tree.py     # Suffix tree implementation
│   └── test_data.py       # Test cases for benchmark suite
├── tests/                 # TODO: Unit tests
└── requirements.txt       # Project dependencies
```

## Usage

### Basic Usage

```bash
python src/main.py data/example1chords.mid "60,64,67" [OPTIONS]
```

### Command Line Arguments

- `midi_file`: Path to the MIDI file to analyze
- `motif`: Comma-separated list of MIDI pitch values representing the melodic motif
- `--delta`: Maximum allowed number of positions that can differ (default: 0)
- `--gamma`: Maximum allowed Sum of Absolute Differences (SAD) between pattern and motif (default: 0)
- `--debug`: Print additional debug information (T_S, loc_map, P)

### Examples

Find occurrences of a C-major triad with at most 1 position differing and SAD ≤ 2:
```bash
python src/main.py data/example1chords.mid "60,62,64" --delta 1 --gamma 2
```

Find exact occurrences of the beginning of "Twinkle Twinkle Little Star":
```bash
python src/main.py data/twinkle.mid "60,60,67,67,69,69,67" --delta 0 --gamma 0
```

## How It Works

1.  The MIDI file is loaded, and melodic parts (channels) are extracted.
2.  Each part is converted into a `tilde{T}` sequence (pitch classes or sets of pitch classes).
3.  `tilde{T}` is converted to its solid equivalent string `T_S` (replacing non-solid sets with unique symbols) and a `loc_map` is created.
4.  The input motif `M` is converted to its solid string `P`.
5.  The combined string `S = T_S #_1 P #_2` is formed.
6.  A suffix tree is built for `S` using Ukkonen's algorithm.
7.  A matching function `M` is defined based on the algorithm's rules and the `loc_map`.
8.  For each potential start position `i` in `T_S`, the `lce_k_gamma_query` method is called on the suffix tree, comparing the suffix starting at `i` against the suffix starting at `P`'s position in `S`, using the matching function `M` and the specified `delta` and `gamma` bounds.
9.  If the query returns a length equal to `P`'s length, an occurrence is recorded at position `i` for that channel.
10. The list of occurrences (channel, position) is returned.

## Advanced Features

- **Non-Solid Symbol Handling**: Correctly handles chords (non-solid symbols) during matching using solid equivalents and a location map, as per the paper.
- **Pitch Class Matching**: Matches are found based on pitch classes (0-11), allowing for octave-invariant matching.
- **Bounded Mismatches**: Allows for specified number of mismatches (`delta`) and a bounded Sum of Absolute Differences (`gamma`) for those mismatches.

## Benchmarking

The project includes a comprehensive benchmarking suite to test the functionality and performance of the motif finder implementation.

### Running Benchmarks

```bash
# Run all benchmarks
python src/benchmark.py

# Run with verbose output
python src/benchmark.py -v

# Run only specific tests
python src/benchmark.py --filter "twinkle"

# Save results to a specific JSON file
python src/benchmark.py --output results/custom_benchmark.json

# Specify a different MIDI folder
python src/benchmark.py --midi-folder custom_data_folder
```

### Benchmark Features

The benchmark suite:
- Runs a comprehensive set of test cases against the motif finder
- Measures execution time for each test
- Tracks memory usage (requires psutil package)
- Validates results against expected outcomes
- Generates detailed reports of test results
- Automatically saves results to the results directory in JSON format for comparison over time

Test cases include various motif patterns, MIDI files, and delta/gamma parameter combinations, from simple exact matches to complex pattern searches with multiple allowed mismatches.

## Complexity Analysis

Two tools are provided to analyze the algorithmic complexity of the implementation:

### 1. Motif Size Complexity Analysis

This tool analyzes how execution time and memory usage scale with increasing motif sizes.

```bash
# Basic usage with default parameters
python src/complexity_analysis.py data/twinkle.mid

# Testing larger motifs with a step size of 2
python src/complexity_analysis.py data/example1chords.mid --start-size 1 --end-size 30 --step 2

# Testing with non-zero delta and gamma
python src/complexity_analysis.py data/twinkle.mid --delta 2 --gamma 4
```

The tool will:
- Generate motifs of increasing sizes
- Measure execution time for each size
- Use curve fitting to estimate the complexity class (e.g., O(n), O(n²), etc.)
- Generate plots showing execution time vs. input size
- Track memory usage (if psutil is installed)
- Save results to results/plots/complexity_analysis.png

### 2. Parameter Complexity Analysis

This tool examines how the delta and gamma parameters affect performance.

```bash
# Analyze both delta and gamma parameters
python src/param_complexity_analysis.py data/twinkle.mid

# Analyze only delta parameter
python src/param_complexity_analysis.py data/example1chords.mid --analyze delta --max-delta 15

# Analyze gamma parameter with fixed delta=2
python src/param_complexity_analysis.py data/twinkle.mid --analyze gamma --fixed-delta 2 --max-gamma 20
```

This analysis helps understand:
- How increasing the delta parameter (allowed pitch mismatches) affects execution time
- How increasing the gamma parameter (allowed SAD) affects execution time
- Whether parameters have linear, quadratic, or exponential effects on performance
- Results are saved as plots to results/plots/delta_complexity.png and results/plots/gamma_complexity.png

Both tools generate visual plots that help in understanding the empirical complexity characteristics of the implementation.

## License

[Your license information here]

## Acknowledgments

- [Reference to the original paper/authors]
- [Any other acknowledgments]