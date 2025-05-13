#!/usr/bin/env python3
import sys, os
# Add this script's directory to PYTHONPATH so local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import argparse
import json
import datetime
import glob
import pandas as pd
from pandas import DataFrame # Import DataFrame directly
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Union, Callable, Optional

# Adjust top-level imports (removed src. prefix)
from motif_finder import MotifFinder
# from midi_processor import MIDIProcessor # May not be needed directly
from test_data import get_test_cases

# Import for memory tracking
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False
    # print("Note: Install 'psutil' package for memory usage tracking") # Keep this commented unless needed

class BenchmarkResult:
    def __init__(self, test_name: str, midi_file: str, motif: List[int], delta: int, gamma: int):
        self.test_name = test_name
        self.midi_file = midi_file
        self.motif = motif
        self.delta = delta
        self.gamma = gamma
        self.occurrences = []
        self.execution_time = 0.0
        self.memory_usage = 0.0  # Memory usage in MB
        self.success = False
        self.error_message = ""
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        result = f"{status} | {self.test_name} | {os.path.basename(self.midi_file)} | "
        result += f"Motif: {self.motif} | δ={self.delta}, γ={self.gamma} | "
        result += f"Found: {len(self.occurrences)} occurrences | "
        result += f"Time: {self.execution_time:.4f}s"

        if self.memory_usage > 0:
            result += f" | Mem: {self.memory_usage:.2f}MB" # Added memory display

        if not self.success:
            result += f" | Error: {self.error_message}" # Added error display

        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_name": self.test_name,
            "midi_file": os.path.basename(self.midi_file),
            "motif": self.motif,
            "delta": self.delta,
            "gamma": self.gamma,
            "occurrences_count": len(self.occurrences),
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "success": self.success,
            "error_message": self.error_message
        }

# --- Beethoven Benchmark Specific Classes and Functions ---

class BeethovenBenchmarkResult:
    """Stores results for a single Beethoven sonata benchmark."""
    def __init__(self, sonata_name: str, delta: int, gamma: int):
        self.sonata_name = sonata_name
        self.delta = delta
        self.gamma = gamma
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.execution_time = 0.0
        self.error_message = ""

    def calculate_metrics(self):
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives

        if (tp + fp) > 0:
            self.precision = tp / (tp + fp)
        else:
            self.precision = 0.0

        if (tp + fn) > 0:
            self.recall = tp / (tp + fn)
        else:
            self.recall = 0.0

        if (self.precision + self.recall) > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

    def __str__(self) -> str:
        status = "✅" if not self.error_message else "❌ ERROR"
        return (f"{status} | {self.sonata_name} | δ={self.delta}, γ={self.gamma} | "
                f"TP:{self.true_positives}, FP:{self.false_positives}, FN:{self.false_negatives} | "
                f"P:{self.precision:.3f}, R:{self.recall:.3f}, F1:{self.f1_score:.3f} | "
                f"Time: {self.execution_time:.4f}s"
                f"{' | Error: ' + self.error_message if self.error_message else ''}")

    def to_dict(self) -> Dict:
        return {
            "sonata_name": self.sonata_name,
            "delta": self.delta,
            "gamma": self.gamma,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
        }

def parse_beethoven_notes(csv_path: str):
    """Parses a Beethoven notes CSV file."""
    try:
        # Define column names based on README
        col_names = ['onset', 'midi', 'morphetic', 'duration', 'staff', 'measure', 'motif_type']
        df = pd.read_csv(csv_path, header=None, names=col_names)
        # Convert relevant columns to numeric, coercing errors
        for col in ['onset', 'midi', 'duration']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where essential numeric conversion failed
        df.dropna(subset=['onset', 'midi', 'duration'], inplace=True)
        # Fill NaN motif types with an empty string or a specific marker
        df['motif_type'] = df['motif_type'].fillna('')
        # Sort by onset time, then potentially by pitch or staff as secondary sort
        df.sort_values(by=['onset', 'staff', 'midi'], inplace=True)
        return df, None
    except Exception as e:
        return None, f"Error parsing {csv_path}: {e}"

def extract_ground_truth_motifs(notes_df):
    """
    Extracts ground truth motif instances from the parsed notes DataFrame.
    Returns a dictionary where keys are motif types and values are lists of motif instances.
    Each motif instance is a list of note dictionaries.
    """
    ground_truth = defaultdict(list)
    current_motif = []
    current_motif_type = None

    for _, note in notes_df.iterrows():
        note_motif_type = note['motif_type'] if pd.notna(note['motif_type']) and note['motif_type'] != '' else None

        if note_motif_type is not None:
            # If starting a new motif or continuing the same type
            if current_motif_type is None or note_motif_type == current_motif_type:
                current_motif.append(note.to_dict())
                current_motif_type = note_motif_type
            # If the type changes, store the completed motif and start a new one
            elif note_motif_type != current_motif_type:
                if current_motif_type is not None and len(current_motif) > 1: # Store only if valid type and length > 1
                    ground_truth[current_motif_type].append(list(current_motif)) # Store a copy
                current_motif = [note.to_dict()]
                current_motif_type = note_motif_type
        else:
            # If the current note is not part of a motif, end the current one
            if current_motif_type is not None and len(current_motif) > 1:
                 ground_truth[current_motif_type].append(list(current_motif))
            current_motif = []
            current_motif_type = None

    # Add the last motif if it exists
    if current_motif_type is not None and len(current_motif) > 1:
        ground_truth[current_motif_type].append(list(current_motif))

    return dict(ground_truth)
    
def parse_beethoven_labels(label_csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parses a Beethoven label CSV file to extract motif start/end times.
    Returns a dict mapping motif type to list of (start_beat, end_beat) tuples.
    """
    df = pd.read_csv(label_csv_path)
    # Convert times (seconds) to beats (1s = 1 beat at 60 QPM)
    df['start_beat'] = df['start_midi']
    df['end_beat'] = df['end_midi']
    grouped = df.groupby('type')[['start_beat','end_beat']]
    label_dict: Dict[str, List[Tuple[float,float]]] = {}
    for motif_type, group in grouped:
        label_dict[motif_type] = [(row.start_beat, row.end_beat) for row in group.itertuples(index=False)]
    return label_dict

def compare_results(detected_motifs, ground_truth_motifs, time_tolerance=0.1):
    """
    Matches detected motifs to ground-truth time windows by motif type and tolerance.

    Args:
        detected_motifs: List of tuples (type, start_beat, end_beat) from detection.
        ground_truth_motifs: Dict mapping motif types to lists of (start_beat, end_beat) from label CSVs.
        time_tolerance: Maximum allowed difference for start and end times to count as a match.

    Returns:
        A tuple (true_positives, false_positives, false_negatives).
    """
    # Real matching: match detected motifs to ground truth time windows by type
    tp = 0
    fp = 0
    matched_gt = set()
    # Iterate detections
    for det in detected_motifs:
        det_type, det_start, det_end = det
        found = False
        # Compare against ground truth of same type
        for idx, (gt_start, gt_end) in enumerate(ground_truth_motifs.get(det_type, [])):
            if (det_type, idx) in matched_gt:
                continue
            if abs(det_start - gt_start) <= time_tolerance and abs(det_end - gt_end) <= time_tolerance:
                tp += 1
                matched_gt.add((det_type, idx))
                found = True
                break
        if not found:
            fp += 1
    # Count ground truth total
    total_gt = sum(len(instances) for instances in ground_truth_motifs.values())
    fn = total_gt - tp
    return tp, fp, fn


def run_beethoven_benchmark_for_sonata(notes_csv_path: str, delta: int, gamma: int) -> BeethovenBenchmarkResult:
    """Runs the benchmark for a single Beethoven sonata CSV file."""
    sonata_name = os.path.basename(notes_csv_path).replace('.csv', '')
    result = BeethovenBenchmarkResult(sonata_name, delta, gamma)
    start_time = time.time()

    try:
        # 1. Parse Notes
        notes_df, error = parse_beethoven_notes(notes_csv_path)
        if error:
            result.error_message = error
            result.execution_time = time.time() - start_time
            return result
        if notes_df is None or notes_df.empty:
             result.error_message = "Parsed notes DataFrame is empty."
             result.execution_time = time.time() - start_time
             return result

        # 2. Load ground truth time windows from label CSVs
        label_csv_path = notes_csv_path.replace('csv_notes', 'csv_label')
        if os.path.exists(label_csv_path):
            ground_truth_windows = parse_beethoven_labels(label_csv_path)
        else:
            ground_truth_windows = {}
            print(f"Warning: Label CSV not found for {sonata_name}, expected at {label_csv_path}")
        # Also get the note-based patterns for each motif type
        note_patterns = extract_ground_truth_motifs(notes_df)  # type: Dict[str, List[List[Dict]]]

        # 3. Run motif detection using naive sliding-window on note sequence
        # Prepare note sequence and pitch classes
        notes_list = notes_df[['onset', 'duration', 'midi']].to_dict('records')
        pc_sequence = [int(n['midi']) % 12 for n in notes_list]
        detected_motifs = []  # List of tuples (motif_type, start_time, end_time)
        # Iterate each motif type and its ground truth time windows
        for motif_type, windows in ground_truth_windows.items():
            # Get the note-based pattern for this motif type (use first instance)
            patterns = note_patterns.get(motif_type, [])
            if not patterns:
                continue
            pattern_notes = patterns[0]
            pattern = [int(note['midi']) % 12 for note in pattern_notes]
            m = len(pattern)
            if m == 0 or len(pc_sequence) < m:
                continue
            # Sliding-window search
            for i in range(len(pc_sequence) - m + 1):
                mismatches = 0
                gamma_sum = 0
                for j in range(m):
                    # Minimum pitch-class difference modulo 12
                    diff = min((pattern[j] - pc_sequence[i+j]) % 12,
                               (pc_sequence[i+j] - pattern[j]) % 12)
                    if diff > 0:
                        mismatches += 1
                        gamma_sum += diff
                        if mismatches > delta or gamma_sum > gamma:
                            break
                if mismatches <= delta and gamma_sum <= gamma:
                    start_time_motif = notes_list[i]['onset']
                    end_time_motif = notes_list[i + m - 1]['onset'] + notes_list[i + m - 1]['duration']
                    detected_motifs.append((motif_type, start_time_motif, end_time_motif))
        # Record detection time
        result.execution_time = time.time() - start_time

        # 4. Compare Results against ground truth windows
        tp, fp, fn = compare_results(detected_motifs, ground_truth_windows)
        result.true_positives = tp
        result.false_positives = fp
        result.false_negatives = fn
        result.calculate_metrics()

    except Exception as e:
        result.error_message = f"Unhandled exception: {e}"
        result.execution_time = time.time() - start_time # Ensure time is recorded on error

    return result


def run_beethoven_benchmark(beethoven_folder: str, delta: int, gamma: int, verbose: bool = False) -> List[BeethovenBenchmarkResult]:
    """Runs the Beethoven benchmark across all sonatas."""
    notes_folder = os.path.join(beethoven_folder, 'csv_notes')
    csv_files = glob.glob(os.path.join(notes_folder, '*.csv'))
    results = []

    if not csv_files:
        print(f"Error: No CSV files found in {notes_folder}")
        return []

    print(f"Found {len(csv_files)} sonata note files in {notes_folder}")
    print(f"Running Beethoven benchmark with δ={delta}, γ={gamma}")
    print("-" * 80)

    for i, csv_path in enumerate(sorted(csv_files)):
        print(f"Processing ({i+1}/{len(csv_files)}): {os.path.basename(csv_path)}...")
        result = run_beethoven_benchmark_for_sonata(csv_path, delta, gamma)
        results.append(result)
        if verbose or result.error_message:
            print(result)

    return results

def print_beethoven_summary(results: List[BeethovenBenchmarkResult]):
    """Prints a summary of the Beethoven benchmark results."""
    if not results:
        print("No Beethoven benchmark results to summarize.")
        return

    total_sonatas = len(results)
    successful_runs = sum(1 for r in results if not r.error_message)
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    total_time = sum(r.execution_time for r in results)

    # Micro-average metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-average metrics (average of per-sonata metrics)
    macro_precision = sum(r.precision for r in results) / successful_runs if successful_runs > 0 else 0.0
    macro_recall = sum(r.recall for r in results) / successful_runs if successful_runs > 0 else 0.0
    macro_f1 = sum(r.f1_score for r in results) / successful_runs if successful_runs > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"BEETHOVEN BENCHMARK SUMMARY ({successful_runs}/{total_sonatas} sonatas processed successfully)")
    print("=" * 80)
    print(f"Parameters: δ={results[0].delta}, γ={results[0].gamma}")
    print(f"Total Execution Time: {total_time:.4f}s")
    print(f"Average Execution Time per Sonata: {total_time / total_sonatas:.4f}s")
    print("-" * 40)
    print("Overall Performance (Micro-Averaged):")
    print(f"  Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
    print(f"  Precision: {micro_precision:.4f}")
    print(f"  Recall:    {micro_recall:.4f}")
    print(f"  F1-Score:  {micro_f1:.4f}")
    print("-" * 40)
    print("Overall Performance (Macro-Averaged):")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print("-" * 40)

    # Print individual results sorted by F1 score (descending)
    print("\nIndividual Sonata Results (Sorted by F1 Score):")
    for result in sorted(results, key=lambda r: r.f1_score, reverse=True):
         print(result)

    # Print errors if any
    errors = [r for r in results if r.error_message]
    if errors:
        print("\n" + "-" * 80)
        print("ERRORS ENCOUNTERED:")
        print("-" * 80)
        for result in errors:
            print(f"{result.sonata_name}: {result.error_message}")


# --- End Beethoven Specific ---


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if has_psutil:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    return 0.0

def run_benchmark_test(test_name: str, midi_file: str, motif: List[int],
                      delta: int = 0, gamma: int = 0,
                      expected_occurrences: int = None) -> BenchmarkResult:
    """
    Run a benchmark test with the given parameters.
    
    Args:
        test_name: Name of the test
        midi_file: Path to the MIDI file
        motif: List of MIDI pitches representing the motif
        delta: Maximum allowed pitch mismatches
        gamma: Maximum allowed Sum of Absolute Differences
        expected_occurrences: Expected number of occurrences (None for no validation)
    
    Returns:
        BenchmarkResult object with test results
    """
    result = BenchmarkResult(test_name, midi_file, motif, delta, gamma)

    try:
        # Get initial memory usage
        initial_memory = get_memory_usage()

        # Initialize motif finder
        # *** NOTE: If MotifFinder is refactored for Beethoven, ensure this still works ***
        # It might need separate initialization paths depending on input type.
        finder = MotifFinder(midi_file) # Assumes MIDI file path init still exists

        # Measure execution time
        start_time = time.time()
        occurrences = finder.find_motif_occurrences(motif, delta, gamma) # Assumes this method still exists
        end_time = time.time()

        # Measure final memory usage
        final_memory = get_memory_usage()

        # Store results
        result.execution_time = end_time - start_time
        result.occurrences = occurrences
        result.memory_usage = max(0, final_memory - initial_memory) # Calculate memory delta

        # Validate results if expected_occurrences is provided
        if expected_occurrences is not None:
            if len(occurrences) == expected_occurrences:
                result.success = True
            else:
                result.success = False
                result.error_message = f"Expected {expected_occurrences}, got {len(occurrences)}"
        else:
            result.success = True # Assume success if no validation needed

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        # Ensure time/memory are recorded even on error
        result.execution_time = time.time() - start_time if 'start_time' in locals() else 0.0
        result.memory_usage = max(0, get_memory_usage() - initial_memory) if 'initial_memory' in locals() else 0.0


    return result

def run_all_benchmarks(midi_folder: str, verbose: bool = False, test_filter: str = None) -> List[BenchmarkResult]:
    """Run all benchmark tests and return the results."""
    results = []
    
    # Get test cases from the test_data module
    test_cases = get_test_cases(midi_folder)
    
    # Filter test cases if filter is provided
    if test_filter:
        test_cases = [tc for tc in test_cases if test_filter.lower() in tc['name'].lower()]
        print(f"Filtered to {len(test_cases)} test cases matching '{test_filter}'")


    # Run all tests
    for i, test_case in enumerate(test_cases):
        print(f"Running test ({i+1}/{len(test_cases)}): {test_case['name']}...")
        result = run_benchmark_test(
            test_name=test_case['name'],
            midi_file=test_case['midi_file'],
            motif=test_case['motif'],
            delta=test_case.get('delta', 0),
            gamma=test_case.get('gamma', 0),
            expected_occurrences=test_case.get('expected')
        )
        results.append(result)
        if verbose or not result.success:
            print(result)


    return results

def print_benchmark_summary(results: List[BenchmarkResult]):
    """Print a summary of benchmark results."""
    if not results:
        print("No standard benchmark results to summarize.") # Added check
        return
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)
    
    total_time = sum(r.execution_time for r in results)
    print(f"Total execution time: {total_time:.4f}s")
    print(f"Average execution time: {total_time/total_tests:.4f}s")
    
    # Print memory stats if available
    if has_psutil and any(r.memory_usage > 0 for r in results):
        total_mem = sum(r.memory_usage for r in results)
        avg_mem = total_mem / total_tests if total_tests > 0 else 0
        max_mem = max(r.memory_usage for r in results) if results else 0
        print(f"Total memory increase: {total_mem:.2f}MB")
        print(f"Average memory increase: {avg_mem:.2f}MB")
        print(f"Max memory increase: {max_mem:.2f}MB")


    # Print results sorted by execution time
    print("\nResults sorted by execution time (fastest to slowest):")
    for i, result in enumerate(sorted(results, key=lambda r: r.execution_time)):
        print(f"{i+1:3d}. {result}") # Added index


    # Print failed tests if any
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        print("\n" + "-" * 80)
        print("FAILED TESTS:")
        print("-" * 80)
        for i, result in enumerate(failed_tests):
            print(f"{i+1}. {result}")


def save_results_to_file(results: List[Union[BenchmarkResult, BeethovenBenchmarkResult]], output_file: str, benchmark_type: str):
    """Save benchmark results to a JSON file."""
    if not results:
        print(f"No results to save for {benchmark_type} benchmark.")
        return

    # Determine structure based on type
    if benchmark_type == "standard":
        total_tests = len(results)
        passed_tests = sum(1 for r in results if isinstance(r, BenchmarkResult) and r.success)
    elif benchmark_type == "beethoven":
        total_tests = len(results)
        # 'Passed' is less relevant here, maybe count successful runs
        passed_tests = sum(1 for r in results if isinstance(r, BeethovenBenchmarkResult) and not r.error_message)
    else:
        print(f"Unknown benchmark type '{benchmark_type}' for saving.")
        return

    output_data = {
        "benchmark_type": benchmark_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_items": total_tests, # Renamed from total_tests for clarity
        "successful_items": passed_tests, # Renamed from passed_tests
        "results": [r.to_dict() for r in results]
    }

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n{benchmark_type.capitalize()} benchmark results saved to {output_file}")
    except Exception as e:
        print(f"\nError saving results to {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark the motif finder implementation')

    # Standard Benchmark Arguments
    standard_group = parser.add_argument_group('Standard Benchmark (using test_data.py)')
    standard_group.add_argument('--midi-folder', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                        help='Path to folder containing MIDI files for standard tests')
    standard_group.add_argument('--filter', type=str, help='Run only standard tests containing this string in their name')

    # Beethoven Benchmark Arguments
    beethoven_group = parser.add_argument_group('Beethoven Benchmark (using Beethoven_motif dataset)')
    beethoven_group.add_argument('--run-beethoven', action='store_true', help='Run the Beethoven dataset benchmark instead of standard tests.')
    beethoven_group.add_argument('--beethoven-folder', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'Beethoven_motif'),
                        help='Path to the root Beethoven_motif dataset folder')
    beethoven_group.add_argument('--beethoven-delta', type=int, default=0, help='Delta value for the Beethoven benchmark')
    beethoven_group.add_argument('--beethoven-gamma', type=int, default=0, help='Gamma value for the Beethoven benchmark')

    # General Arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output for each test/sonata')
    parser.add_argument('--output', '-o', type=str, help='Save results to specified JSON file (default: results/benchmark_<type>_<timestamp>.json)')

    args = parser.parse_args()

    # Determine output file path
    output_file = args.output
    if output_file is None:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_type_tag = "beethoven" if args.run_beethoven else "standard"
        output_file = os.path.join(results_dir, f"benchmark_{benchmark_type_tag}_{timestamp}.json")
        print(f"Output file not specified, will save to: {output_file}")


    if args.run_beethoven:
        # --- Run Beethoven Benchmark ---
        print(f"Starting Beethoven benchmark using data from {args.beethoven_folder}")
        print("-" * 80)
        if not os.path.isdir(args.beethoven_folder):
             print(f"Error: Beethoven folder not found at {args.beethoven_folder}")
             sys.exit(1)
        if not os.path.isdir(os.path.join(args.beethoven_folder, 'csv_notes')):
             print(f"Error: 'csv_notes' subfolder not found in {args.beethoven_folder}")
             sys.exit(1)

        # Check for pandas (already imported at top level)
        if pd is None:
            print("Error: 'pandas' package is required for the Beethoven benchmark. Please install it (`pip install pandas`).")
            sys.exit(1)

        results = run_beethoven_benchmark(args.beethoven_folder, args.beethoven_delta, args.beethoven_gamma, args.verbose)
        print_beethoven_summary(results)
        save_results_to_file(results, output_file, "beethoven")

    else:
        # --- Run Standard Benchmark ---
        print(f"Starting standard benchmark using MIDI files from {args.midi_folder}")
        print("-" * 80)
        if not os.path.isdir(args.midi_folder):
             print(f"Error: MIDI folder not found at {args.midi_folder}")
             sys.exit(1)

        results = run_all_benchmarks(args.midi_folder, args.verbose, args.filter)
        print_benchmark_summary(results)
        save_results_to_file(results, output_file, "standard")


if __name__ == "__main__":
    # Imports needed specifically for the main guard execution context
    import sys
    import os
    # Other imports like argparse, datetime, json, glob, defaultdict are already top-level

    # Ensure pandas is available (check again in case top-level failed silently)
    try:
        import pandas as pd
        from pandas import DataFrame
    except ImportError:
        pd = None
        DataFrame = None # Define as None if pandas is missing

    # Ensure psutil is available
    try:
        import psutil
        has_psutil = True # Ensure this is set if import succeeds here
    except ImportError:
        has_psutil = False # Ensure this is False if import fails here

    # Add project root/src to sys.path to allow direct imports
    # Correctly find project root assuming benchmark.py is in src/
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(src_dir)

    # Add project_root first, then src_dir? Or just src_dir?
    # Let's add project_root so imports relative to the root *might* work if needed elsewhere,
    # and src_dir for direct imports like `from motif_finder ...`
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir) # Add src dir itself for direct imports

    # Now, re-attempt imports within a try-except to give a clear error if modules are missing
    try:
        from motif_finder import MotifFinder
        from test_data import get_test_cases
        # midi_processor might be needed depending on refactoring
        # from midi_processor import MIDIProcessor
    except ImportError as e:
        print(f"Error importing project modules: {e}")
        print(f"Python Path: {sys.path}")
        print("Ensure the script is run correctly (e.g., `python src/benchmark.py`) and all modules exist.")
        sys.exit(1)

    main()