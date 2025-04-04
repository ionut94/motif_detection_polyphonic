#!/usr/bin/env python3
"""
Complexity Analysis Tool for the Motif Finder implementation.

This script measures how execution time scales with different input parameters
to estimate the algorithmic complexity of the implementation.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Set
import matplotlib.pyplot as plt
import cProfile
import pstats
import io

# Add the src directory to the Python path FIRST
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules from src
from src.suffix_tree import SuffixTree
from src.motif_finder import MotifFinder, SEPARATOR_1, SEPARATOR_2
from src.midi_processor import MIDIProcessor

# Try to import for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install 'psutil' package for memory usage tracking")

class ComplexityMeasurement:
    def __init__(self, input_size: int, execution_time: float, memory_usage: float = 0):
        self.input_size = input_size
        self.execution_time = execution_time
        self.memory_usage = memory_usage
    
    def __str__(self) -> str:
        result = f"Input size: {self.input_size}, Time: {self.execution_time:.6f}s"
        if self.memory_usage > 0:
            result += f", Memory: {self.memory_usage:.2f}MB"
        return result

def generate_motifs(start_size: int, end_size: int, step: int = 1) -> Dict[int, List[int]]:
    """Generate motifs of different sizes for testing complexity."""
    motifs = {}
    # Use C major scale pattern for predictable motifs
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    
    for size in range(start_size, end_size + 1, step):
        # Generate a motif of specified size by repeating the scale pattern
        motif = []
        for i in range(size):
            motif.append(scale[i % len(scale)])
        motifs[size] = motif
    
    return motifs

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    return 0.0

def measure_execution_time(midi_file: str, motif: List[int], delta: int, gamma: int) -> Tuple[float, float]:
    """
    Measure execution time and memory usage for finding a motif in a MIDI file.
    
    Args:
        midi_file: Path to the MIDI file
        motif: The motif to search for
        delta: The delta parameter
        gamma: The gamma parameter
    
    Returns:
        Tuple of (execution_time, memory_usage)
    """
    # Get initial memory usage
    initial_memory = get_memory_usage()
    
    # Initialize motif finder
    finder = MotifFinder(midi_file)
    
    # Measure execution time
    start_time = time.time()
    finder.find_motif_occurrences(motif, delta, gamma)
    end_time = time.time()
    
    # Measure final memory usage
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory if initial_memory > 0 else 0
    
    return end_time - start_time, memory_used

def estimate_complexity_class(sizes: List[int], times: List[float]) -> str:
    """Enhanced complexity estimation with more models and better statistical fit."""
    if len(sizes) < 3:
        return "Need more data points"

    x = np.array(sizes, dtype=float)
    y = np.array(times, dtype=float)

    # Filter out zero sizes for log-based complexities
    valid_indices_log = x > 0
    x_log = x[valid_indices_log]
    y_log = y[valid_indices_log]

    fits = {}
    errors = {}

    try: # O(1)
        params = [np.mean(y)]
        fits["O(1)"] = params
        errors["O(1)"] = np.sum((params[0] - y)**2)
    except Exception: errors["O(1)"] = float('inf')

    try: # O(log n)
        if len(x_log) >= 2:
            A = np.vstack([np.log(x_log), np.ones(len(x_log))]).T
            params, res, _, _ = np.linalg.lstsq(A, y_log, rcond=None)
            fits["O(log n)"] = params
            errors["O(log n)"] = np.sum((A.dot(params) - y_log)**2) if len(res) == 0 else res[0]
        else: errors["O(log n)"] = float('inf')
    except Exception: errors["O(log n)"] = float('inf')

    try: # O(n)
        A = np.vstack([x, np.ones(len(x))]).T
        params, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        fits["O(n)"] = params
        errors["O(n)"] = np.sum((A.dot(params) - y)**2) if len(res) == 0 else res[0]
    except Exception: errors["O(n)"] = float('inf')

    try: # O(n log n)
        if len(x_log) >= 2:
            A = np.vstack([x_log * np.log(x_log), np.ones(len(x_log))]).T
            params, res, _, _ = np.linalg.lstsq(A, y_log, rcond=None)
            fits["O(n log n)"] = params
            errors["O(n log n)"] = np.sum((A.dot(params) - y_log)**2) if len(res) == 0 else res[0]
        else: errors["O(n log n)"] = float('inf')
    except Exception: errors["O(n log n)"] = float('inf')

    try: # O(n^2)
        A = np.vstack([x**2, np.ones(len(x))]).T
        params, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        fits["O(n²)"] = params
        errors["O(n²)"] = np.sum((A.dot(params) - y)**2) if len(res) == 0 else res[0]
    except Exception: errors["O(n²)"] = float('inf')

    try: # O(n^1.5)
        A = np.vstack([x**1.5, np.ones(len(x))]).T
        params, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        fits["O(n^1.5)"] = params
        errors["O(n^1.5)"] = np.sum((A.dot(params) - y)**2) if len(res) == 0 else res[0]
    except Exception: errors["O(n^1.5)"] = float('inf')
    
    # Normalize errors by degrees of freedom for fair comparison
    normalized_errors = {k: v/(len(sizes)-len(fits[k])) if len(sizes) > len(fits[k]) else float('inf') 
                        for k, v in errors.items()}
    best_fit = min(normalized_errors, key=normalized_errors.get) if normalized_errors else "Unknown"

    # Basic check for exponential - if O(n^2) error is still high and time increases rapidly
    if best_fit in ["O(n)", "O(n log n)", "O(n²)"] and errors[best_fit] > 1e-4: # Arbitrary threshold
         if len(y) > 2 and y[-1] / y[-2] > 1.5 and y[-2] / y[-3] > 1.5: # Rapid increase check
              # This is a very rough heuristic for exponential
              # A proper fit for O(2^n) is more complex
              pass # Could add O(2^n) fit here if needed

    return best_fit

def analyze_motif_size_complexity(midi_file: str, start_size: int = 1, end_size: int = 20, 
                                 step: int = 1, delta: int = 0, gamma: int = 0,
                                 repetitions: int = 3, plot: bool = True) -> List[ComplexityMeasurement]:
    """
    Analyze how execution time scales with motif size.
    
    Args:
        midi_file: Path to the MIDI file
        start_size: Minimum motif size to test
        end_size: Maximum motif size to test
        step: Step size between motif sizes
        delta: Delta parameter for all tests
        gamma: Gamma parameter for all tests
        repetitions: Number of times to repeat each measurement (for more accurate timing)
        plot: Whether to generate plots
        
    Returns:
        List of ComplexityMeasurement objects
    """
    print(f"Analyzing complexity with motif sizes from {start_size} to {end_size} (step {step})")
    print(f"Using parameters: delta={delta}, gamma={gamma}")
    print(f"Repeating each measurement {repetitions} times for accuracy")
    print("-" * 80)
    
    motifs = generate_motifs(start_size, end_size, step)
    measurements = []
    
    for size, motif in motifs.items():
        print(f"Testing motif size {size}... ", end="", flush=True)
        
        # Run multiple measurements and take the average (to reduce noise)
        times = []
        memory_usages = []
        
        for _ in range(repetitions):
            time_taken, memory_used = measure_execution_time(midi_file, motif, delta, gamma)
            times.append(time_taken)
            memory_usages.append(memory_used)
        
        # Take the average
        avg_time = sum(times) / len(times)
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        
        measurement = ComplexityMeasurement(size, avg_time, avg_memory)
        measurements.append(measurement)
        
        print(f"Avg time: {avg_time:.6f}s")
    
    # Extract sizes and times for analysis
    sizes = [m.input_size for m in measurements]
    times = [m.execution_time for m in measurements]
    memory_usages = [m.memory_usage for m in measurements]
    
    # Estimate complexity class
    complexity_class = estimate_complexity_class(sizes, times)
    print(f"\nEstimated complexity class: {complexity_class}")
    
    if plot:
        plot_complexity(sizes, times, memory_usages, complexity_class, analysis_type='motif')
    
    return measurements

def plot_complexity(sizes, times, memory_usages, complexity_class, analysis_type: str):
    """Generate plots for time and memory complexity."""
    plt.figure(figsize=(12, 10))
    input_label = 'Motif Length (m)' if analysis_type == 'motif' else 'Text Length (n)'
    plot_filename = 'motif_complexity.png' if analysis_type == 'motif' else 'text_complexity.png'

    # Time complexity plot
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=6) # Smaller markers
    plt.title(f'Time Complexity vs {input_label} (Estimated: {complexity_class})')
    plt.xlabel(input_label)
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.yscale('log') # Use log scale for time if values vary widely
    plt.xscale('log') # Use log scale for size if steps are large/logarithmic

    # Memory usage plot if available
    if HAS_PSUTIL and any(m > 0 for m in memory_usages):
        plt.subplot(2, 1, 2)
        plt.plot(sizes, memory_usages, 'o-', color='green', linewidth=2, markersize=6)
        plt.title(f'Memory Usage vs {input_label}')
        plt.xlabel(input_label)
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xscale('log') # Match x-scale

    plt.tight_layout(pad=3.0) # Add padding

    # Ensure results/plots directory exists
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
    os.makedirs(results_dir, exist_ok=True)

    # Save the figure
    output_file = os.path.join(results_dir, plot_filename)
    try:
        plt.savefig(output_file)
        print(f"\nPlot saved as '{output_file}'")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    # Optionally show plot
    # try:
    #     plt.show()
    # except Exception as e:
    #     print(f"Note: Could not display plot interactively ({e})")
    plt.close() # Close the figure window


# --- New Function for Text Size Analysis ---

def measure_text_size_execution(T_S_prefix: str, loc_map_prefix: Dict[str, Set[int]],
                                P: str, delta: int, gamma: int,
                                finder_instance: MotifFinder) -> Tuple[float, float]:
    """
    Measures execution time for a given T_S prefix and fixed motif P.
    Reuses the finder instance for its helper methods but performs core logic here.
    """
    initial_memory = get_memory_usage()
    start_time = time.time()

    n = len(T_S_prefix)
    m = len(P)
    if n < m:
        return 0.0, 0.0 # Cannot find motif if text is shorter

    # 1. Create combined string S
    S = T_S_prefix + SEPARATOR_1 + P + SEPARATOR_2
    start_index_P = n + 1

    # 2. Create matching function (using finder's method for consistency)
    match_function = finder_instance._create_matching_function(loc_map_prefix, gamma)

    # 3. Build Suffix Tree for S
    try:
        suffix_tree = SuffixTree(S)
    except Exception as e:
        print(f"\nError building suffix tree for n={n}: {e}")
        return -1.0, 0.0 # Indicate error

    # 4. Perform LCE queries
    try:
        for i in range(n - m + 1):
            # We only care about the time, not the result for complexity analysis
            _ = suffix_tree.lce_k_gamma_query(
                i, start_index_P, delta, gamma, match_function
            )
    except Exception as e:
         print(f"\nError during LCE query for n={n}, i={i}: {e}")
         return -1.0, 0.0 # Indicate error

    end_time = time.time()
    final_memory = get_memory_usage()
    memory_used = final_memory - initial_memory if initial_memory > 0 else 0

    return end_time - start_time, memory_used


def analyze_text_size_complexity(midi_file: str, fixed_motif: List[int],
                                 num_steps: int = 10, channel_to_analyze: Optional[int] = None,
                                 delta: int = 0, gamma: int = 0,
                                 repetitions: int = 3, plot: bool = True) -> List[ComplexityMeasurement]:
    """
    Analyze how execution time scales with text size (n).
    """
    print(f"Analyzing complexity with text size (n) using up to {num_steps} steps.")
    print(f"Using fixed motif: {fixed_motif} (length {len(fixed_motif)})")
    print(f"Using parameters: delta={delta}, gamma={gamma}")
    print(f"Repeating each measurement {repetitions} times for accuracy")
    print("-" * 80)

    # 1. Process MIDI once to get full T_S and loc_map
    try:
        processor = MIDIProcessor(midi_file)
        # Choose the channel to analyze (e.g., the longest one)
        if channel_to_analyze is None:
            target_channel = max(processor.tilde_T_parts, key=lambda k: len(processor.tilde_T_parts[k]), default=None)
            if target_channel is None:
                print("Error: No channels found in MIDI file.")
                return []
            print(f"Automatically selected longest channel: {target_channel}")
        else:
            if channel_to_analyze not in processor.tilde_T_parts:
                print(f"Error: Channel {channel_to_analyze} not found in MIDI file.")
                return []
            target_channel = channel_to_analyze
            print(f"Using specified channel: {target_channel}")

        tilde_T_full = processor.get_tilde_T(target_channel)
        if not tilde_T_full:
            print(f"Error: Channel {target_channel} is empty.")
            return []

        T_S_full, loc_map_full = MIDIProcessor.create_solid_equivalent(tilde_T_full)
        n_full = len(T_S_full)
        print(f"Full text length (n) for channel {target_channel}: {n_full}")

        # Preprocess the fixed motif
        P = MIDIProcessor.preprocess_motif(fixed_motif)
        m = len(P)
        if m == 0:
            print("Error: Fixed motif is empty after preprocessing.")
            return []

        # Create a MotifFinder instance with the actual MIDI file path
        finder_instance = MotifFinder(midi_file)  # Use the real MIDI file path
        finder_instance.midi_processor = processor  # Link the processor we already created

    except Exception as e:
        print(f"Error during initial processing: {e}")
        return []

    measurements = []
    tested_sizes = set()

    # 2. Loop through different text lengths (n) - using percentage steps
    for i in range(1, num_steps + 1):
        n = int(n_full * (i / num_steps))
        if n < m or n == 0 or n in tested_sizes: # Ensure n >= m and avoid duplicates
            continue
        tested_sizes.add(n)

        print(f"Testing text size n = {n}... ", end="", flush=True)

        # Create prefix of T_S
        T_S_prefix = T_S_full[:n]

        # Create corresponding loc_map prefix (only include symbols present in T_S_prefix)
        loc_map_prefix = {char: pitch_set for char, pitch_set in loc_map_full.items() if char in T_S_prefix}

        # Run multiple measurements
        times = []
        memory_usages = []
        error_occurred = False
        for rep in range(repetitions):
            time_taken, memory_used = measure_text_size_execution(
                T_S_prefix, loc_map_prefix, P, delta, gamma, finder_instance
            )
            if time_taken < 0: # Error indicator
                 error_occurred = True
                 break
            times.append(time_taken)
            memory_usages.append(memory_used)

        if error_occurred:
             print("Error during measurement, skipping size.")
             continue

        # Average results
        avg_time = sum(times) / len(times) if times else 0
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0

        measurement = ComplexityMeasurement(n, avg_time, avg_memory)
        measurements.append(measurement)
        print(f"Avg time: {avg_time:.6f}s")

    if not measurements:
         print("\nNo valid measurements taken.")
         return []

    # 3. Analyze and plot results
    sizes = [m.input_size for m in measurements]
    times = [m.execution_time for m in measurements]
    memory_usages = [m.memory_usage for m in measurements]

    complexity_class = estimate_complexity_class(sizes, times)
    print(f"\nEstimated complexity class vs text length (n): {complexity_class}")

    if plot:
        plot_complexity(sizes, times, memory_usages, complexity_class, analysis_type='text')

    return measurements


# --- New Function for Parameter Sensitivity Analysis ---

def analyze_parameter_sensitivity(midi_file: str, motif: List[int], 
                                 parameter: str, max_value: int = 5,
                                 repetitions: int = 3) -> None:
    """Analyze how delta or gamma affects performance with fixed motif and text."""
    print(f"Analyzing {parameter} sensitivity from 0 to {max_value}")
    
    times = []
    for value in range(max_value + 1):
        delta = value if parameter == 'delta' else 0
        gamma = value if parameter == 'gamma' else 0
        
        print(f"Testing {parameter}={value}... ", end="", flush=True)
        
        # Run measurements
        run_times = []
        for _ in range(repetitions):
            time_taken, _ = measure_execution_time(midi_file, motif, delta, gamma)
            run_times.append(time_taken)
        
        avg_time = sum(run_times) / len(run_times)
        times.append(avg_time)
        print(f"Avg time: {avg_time:.6f}s")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_value + 1), times, marker='o', linestyle='-')
    plt.title(f'Execution Time vs {parameter.capitalize()} Value')
    plt.xlabel(f'{parameter.capitalize()} Value')
    plt.ylabel('Execution Time (s)')
    plt.grid(True)
    
    # Save the plot
    output_file = f'{parameter}_sensitivity_{time.strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')
    
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display plot interactively ({e})")
    plt.close()

def expected_best_fit(analysis_type: str) -> str:
    """Return the theoretically expected complexity class for the given analysis type."""
    if (analysis_type == 'motif'):
        return "O(n)"  # For fixed text and varying motif length
    elif (analysis_type == 'text'):
        return "O(n)"  # For fixed motif and varying text length
    else:
        return "Unknown"

def compare_with_theory(sizes, times, best_fit, analysis_type):
    """Compare empirical results with theoretical complexity."""
    print("\n=== Theoretical vs Empirical Comparison ===")
    
    if analysis_type == 'motif':
        print("Theory: Suffix tree construction is O(n), LCE query is O(m)")
        print("Expected: O(n + m*q) where q is the number of queries")
    else:  # text analysis
        print("Theory: Suffix tree construction is O(n+m), LCE query is O(1)")
        print("Expected: O(n+m) for construction + O(n-m+1) for queries")
    
    if best_fit == expected_best_fit(analysis_type):
        print("MATCH: Empirical results match theoretical expectations")
    else:
        print(f"MISMATCH: Expected {expected_best_fit(analysis_type)} but found {best_fit}")
        print("Possible reasons:")
        print("1. Constants dominate for the input sizes tested")
        print("2. Implementation details affecting performance")
        print("3. Memory allocation/garbage collection overhead")

def profile_execution(func, *args, **kwargs):
    """Profile a function execution and return results."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Top 15 functions
    return result, s.getvalue()

def run_all_analyses(midi_file: str, args):
    """Run all available analyses sequentially."""
    print("\n" + "="*80)
    print(" RUNNING COMPREHENSIVE COMPLEXITY ANALYSIS ".center(80, "="))
    print("="*80 + "\n")
    
    # 1. Motif size complexity
    print("\n" + "="*30 + " MOTIF SIZE ANALYSIS " + "="*30)
    motif_measurements = analyze_motif_size_complexity(
        midi_file,
        args.start_m,
        args.end_m,
        args.step_m,
        args.delta,
        args.gamma,
        args.repetitions,
        not args.no_plot
    )
    if motif_measurements:
        sizes = [m.input_size for m in motif_measurements]
        times = [m.execution_time for m in motif_measurements]
        complexity_class = estimate_complexity_class(sizes, times)
        compare_with_theory(sizes, times, complexity_class, 'motif')
    
    # 2. Text size complexity
    print("\n" + "="*30 + " TEXT SIZE ANALYSIS " + "="*31)
    try:
        fixed_motif_list = [int(p.strip()) for p in args.fixed_motif.split(',')]
        text_measurements = analyze_text_size_complexity(
            midi_file,
            fixed_motif_list,
            args.steps_n,
            args.channel,
            args.delta,
            args.gamma,
            args.repetitions,
            not args.no_plot
        )
        if text_measurements:
            sizes = [m.input_size for m in text_measurements]
            times = [m.execution_time for m in text_measurements]
            complexity_class = estimate_complexity_class(sizes, times)
            compare_with_theory(sizes, times, complexity_class, 'text')
    except ValueError as e:
        print(f"Error with text analysis: {e}")
    
    # 3. Delta parameter sensitivity
    print("\n" + "="*30 + " DELTA SENSITIVITY ANALYSIS " + "="*23)
    try:
        motif = [int(x) for x in args.fixed_motif.split(',')]
        analyze_parameter_sensitivity(
            midi_file, motif, 'delta', args.max_delta, args.repetitions
        )
    except ValueError as e:
        print(f"Error with delta sensitivity analysis: {e}")
    
    # 4. Gamma parameter sensitivity
    print("\n" + "="*30 + " GAMMA SENSITIVITY ANALYSIS " + "="*23)
    try:
        motif = [int(x) for x in args.fixed_motif.split(',')]
        analyze_parameter_sensitivity(
            midi_file, motif, 'gamma', args.max_gamma, args.repetitions
        )
    except ValueError as e:
        print(f"Error with gamma sensitivity analysis: {e}")
    
    print("\n" + "="*80)
    print(" COMPREHENSIVE ANALYSIS COMPLETE ".center(80, "="))
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze the algorithmic complexity of motif finder.')
    parser.add_argument('midi_file', help='Path to the MIDI file for analysis.')
    parser.add_argument('--mode', choices=['motif', 'text'], default='motif',
                        help='Analysis mode: "motif" (vary motif length m) or "text" (vary text length n).')
    
    # Add the run-all option
    parser.add_argument('--run-all', action='store_true',
                        help='Run all analysis modes sequentially')
    
    # Rest of your existing arguments
    parser.add_argument('--start-m', type=int, default=1, help='[Motif Mode] Starting motif size (m).')
    parser.add_argument('--end-m', type=int, default=20, help='[Motif Mode] Ending motif size (m).')
    parser.add_argument('--step-m', type=int, default=1, help='[Motif Mode] Step size for motif length.')
    parser.add_argument('--fixed-motif', type=str, default="60,67,72", # Default C-G-C'
                        help='[Text Mode] Fixed motif (comma-separated MIDI pitches) to use.')
    parser.add_argument('--steps-n', type=int, default=10,
                        help='[Text Mode] Number of text length steps (percentages) to analyze.')
    parser.add_argument('--channel', type=int, default=None,
                        help='[Text Mode] Specific MIDI channel to analyze (default: longest).')
    parser.add_argument('--delta', type=int, default=0, help='Delta parameter for all tests.')
    parser.add_argument('--gamma', type=int, default=0, help='Gamma parameter for all tests.')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of repetitions per measurement.')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting.')
    parser.add_argument('--vary-delta', action='store_true', 
                        help='Analyze how delta parameter affects performance')
    parser.add_argument('--vary-gamma', action='store_true',
                        help='Analyze how gamma parameter affects performance')
    parser.add_argument('--max-delta', type=int, default=5,
                        help='Maximum delta value to test')
    parser.add_argument('--max-gamma', type=int, default=10,
                        help='Maximum gamma value to test')
    parser.add_argument('--profile', action='store_true', 
                        help='Enable detailed performance profiling')
    
    args = parser.parse_args()
    
    # Check for run-all mode first
    if args.run_all:
        run_all_analyses(args.midi_file, args)
        return
    
    # Check profile mode first to wrap other modes
    if args.profile:
        print("Performance profiling enabled")
        if args.vary_delta:
            try:
                motif = [int(x) for x in args.fixed_motif.split(',')]
            except ValueError:
                print("Error: --fixed-motif must be comma-separated integers.")
                sys.exit(1)
            _, profile_stats = profile_execution(
                analyze_parameter_sensitivity,
                args.midi_file, motif, 'delta', args.max_delta, args.repetitions
            )
            print("\n=== Performance Profile ===")
            print(profile_stats)
        elif args.vary_gamma:
            try:
                motif = [int(x) for x in args.fixed_motif.split(',')]
            except ValueError:
                print("Error: --fixed-motif must be comma-separated integers.")
                sys.exit(1)
            _, profile_stats = profile_execution(
                analyze_parameter_sensitivity,
                args.midi_file, motif, 'gamma', args.max_gamma, args.repetitions
            )
            print("\n=== Performance Profile ===")
            print(profile_stats)
        elif args.mode == 'motif':
            _, profile_stats = profile_execution(
                analyze_motif_size_complexity,
                args.midi_file, args.start_m, args.end_m, args.step_m,
                args.delta, args.gamma, args.repetitions, not args.no_plot
            )
            print("\n=== Performance Profile ===")
            print(profile_stats)
        elif args.mode == 'text':
            try:
                fixed_motif_list = [int(p.strip()) for p in args.fixed_motif.split(',')]
            except ValueError:
                print("Error: --fixed-motif must be comma-separated integers.")
                sys.exit(1)
            _, profile_stats = profile_execution(
                analyze_text_size_complexity,
                args.midi_file, fixed_motif_list, args.steps_n, args.channel,
                args.delta, args.gamma, args.repetitions, not args.no_plot
            ) 
            print("\n=== Performance Profile ===")
            print(profile_stats)
        return
        
    # Regular execution modes
    # Check special modes first
    if args.vary_delta:
        try:
            motif = [int(x) for x in args.fixed_motif.split(',')]
        except ValueError:
            print("Error: --fixed-motif must be comma-separated integers.")
            sys.exit(1)
        analyze_parameter_sensitivity(
            args.midi_file, motif, 'delta', args.max_delta, args.repetitions
        )
    elif args.vary_gamma:
        try:
            motif = [int(x) for x in args.fixed_motif.split(',')]
        except ValueError:
            print("Error: --fixed-motif must be comma-separated integers.")
            sys.exit(1)
        analyze_parameter_sensitivity(
            args.midi_file, motif, 'gamma', args.max_gamma, args.repetitions
        )
    # Then check standard modes
    elif args.mode == 'motif':
        measurements = analyze_motif_size_complexity(
            args.midi_file,
            args.start_m,
            args.end_m,
            args.step_m,
            args.delta,
            args.gamma,
            args.repetitions,
            not args.no_plot
        )
        # Compare with theoretical expectations if we have measurements
        if measurements:
            sizes = [m.input_size for m in measurements]
            times = [m.execution_time for m in measurements]
            complexity_class = estimate_complexity_class(sizes, times)
            compare_with_theory(sizes, times, complexity_class, 'motif')
    elif args.mode == 'text':
        try:
            fixed_motif_list = [int(p.strip()) for p in args.fixed_motif.split(',')]
        except ValueError:
            print("Error: --fixed-motif must be comma-separated integers.")
            sys.exit(1)

        measurements = analyze_text_size_complexity(
            args.midi_file,
            fixed_motif_list,
            args.steps_n,
            args.channel,
            args.delta,
            args.gamma,
            args.repetitions,
            not args.no_plot
        )
        # Compare with theoretical expectations if we have measurements
        if measurements:
            sizes = [m.input_size for m in measurements]
            times = [m.execution_time for m in measurements]
            complexity_class = estimate_complexity_class(sizes, times)
            compare_with_theory(sizes, times, complexity_class, 'text')


if __name__ == "__main__":
    main()