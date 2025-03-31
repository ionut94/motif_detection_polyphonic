# Melodic Motif Finder

A Python implementation of an algorithm for finding melodic motifs in MIDI files with bounded mismatches.

## Overview

This project implements an algorithm for efficiently finding melodic motifs in MIDI files, as described in the academic paper. The algorithm allows for bounded mismatches in both the number of different positions (delta) and the Sum of Absolute Differences (gamma), making it robust for musical pattern matching where exact matches might be rare.

## Algorithm

The implementation follows an algorithm described in the paper. Given a MIDI file T containing tuples in the form (message, pitch, keyvelocity, channel, timestamp), and a melodic motif M of length m, each of the motif occurrences with delta, gamma bounded mismatches can be found as follows:

1. **Extract each melodic part and convert to degenerate strings**: 
   Each melodic part is extracted from the MIDI file and converted into a degenerate string X_{$} where chords are represented as non-solid symbols (multiple pitch options).

2. **Convert the motif M into a string P**:
   The melodic motif is converted to a string representation.

3. **Create a combined string S**:
   For each degenerate string X_{$}, create a new string S = X_{$}#_{1}P#_{2} along with a corresponding matching table, where #_{1} and #_{2} are special separator characters.

4. **Construct the suffix tree**:
   For each string S, construct the unordered suffix tree in linear time.

5. **Perform LCE queries**:
   For each string S, perform n LCE_{delta} (S, i, n) queries.
   - For each mismatch, check if it's within the delta, gamma bounds.
   - If the mismatch is between a non-solid symbol and a solid symbol, identify the character which has the smallest difference with the queried character in P using the formula: min((x≠y) mod 12, (y≠x) mod 12)

6. **Identify occurrences**:
   After each LCE_{delta+k} query, if the string returned has length m, then it is returned as a motif occurrence.

## Requirements

- Python 3.6+
- mido library (for MIDI file processing)
- numpy
- matplotlib (for analysis and visualization)
- psutil (optional, for memory usage tracking)

## Installation

1. Clone this repository:
   ```
   git clone https://your-repo-url/music_paper.git
   cd music_paper
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
├── docs/                  # Documentation files
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
├── tests/                 # Unit tests
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
- `--delta`: Maximum allowed number of positions that can differ (default: 1)
- `--gamma`: Maximum allowed Sum of Absolute Differences (SAD) between pattern and motif (default: 0)
- `--debug`: Print additional debug information

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

1. The MIDI file is loaded and analyzed to extract melodic parts by channel.
2. Each melodic part is converted into a degenerate string representation where:
   - Single notes are represented as characters (solid symbols)
   - Chords are represented as sets of characters enclosed in brackets (non-solid symbols)
3. The motif is converted to a string representation.
4. For each melodic part, a suffix tree is constructed to efficiently find pattern occurrences.
5. LCE queries with bounded mismatches are performed to identify potential motif occurrences.
6. Occurrences are verified against the delta and gamma bounds.
7. The list of motif occurrences is returned, showing the channel and position of each match.

## Advanced Features

- **Degenerate String Support**: The algorithm handles both single notes and chords (represented as degenerate strings).
- **Pitch Class Matching**: Matches are found based on pitch classes (0-11), allowing for octave-invariant matching.
- **Bounded Mismatches**: Allows for specified number of mismatches in both pitch (delta) and rhythm (gamma).

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