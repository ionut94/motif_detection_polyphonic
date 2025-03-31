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
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motif_finder import MotifFinder
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
    """
    Estimate the complexity class by fitting different curves to the data.
    
    Args:
        sizes: List of input sizes
        times: List of corresponding execution times
    
    Returns:
        String describing the likely complexity class
    """
    if len(sizes) < 3:
        return "Need more data points to estimate complexity"
    
    # Convert to numpy arrays
    x = np.array(sizes)
    y = np.array(times)
    
    # Define functions to fit
    complexity_classes = {
        "O(1)": lambda x, a: a * np.ones_like(x),
        "O(log n)": lambda x, a, b: a * np.log(x) + b,
        "O(n)": lambda x, a, b: a * x + b,
        "O(n log n)": lambda x, a, b: a * x * np.log(x) + b,
        "O(n²)": lambda x, a, b: a * x**2 + b,
        "O(n³)": lambda x, a, b: a * x**3 + b,
        "O(2^n)": lambda x, a, b: a * np.power(2, x) + b
    }
    
    best_fit = None
    best_error = float('inf')
    best_params = None
    
    # Try fitting each function
    for name, func in complexity_classes.items():
        try:
            if name == "O(1)":
                params, residuals = np.polyfit(np.ones_like(x), y, 0, full=True)[:2]
                error = residuals[0] if len(residuals) > 0 else float('inf')
            else:
                # For other functions, we'll use a simple approach to estimate parameters
                if name in ["O(log n)", "O(n log n)"]:
                    # Avoid log(0)
                    valid_indices = x > 0
                    if np.sum(valid_indices) < 2:
                        continue
                    x_valid = x[valid_indices]
                    y_valid = y[valid_indices]
                else:
                    x_valid = x
                    y_valid = y
                    
                if name == "O(log n)":
                    X = np.column_stack((np.log(x_valid), np.ones_like(x_valid)))
                elif name == "O(n)":
                    X = np.column_stack((x_valid, np.ones_like(x_valid)))
                elif name == "O(n log n)":
                    X = np.column_stack((x_valid * np.log(x_valid), np.ones_like(x_valid)))
                elif name == "O(n²)":
                    X = np.column_stack((x_valid**2, np.ones_like(x_valid)))
                elif name == "O(n³)":
                    X = np.column_stack((x_valid**3, np.ones_like(x_valid)))
                elif name == "O(2^n)":
                    X = np.column_stack((np.power(2, x_valid), np.ones_like(x_valid)))
                
                # Use least squares to estimate parameters
                params, residuals, _, _ = np.linalg.lstsq(X, y_valid, rcond=None)
                error = np.sum((np.dot(X, params) - y_valid)**2)
            
            if error < best_error:
                best_fit = name
                best_error = error
                best_params = params
                
        except Exception as e:
            print(f"Error fitting {name}: {e}")
    
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
        plot_complexity(sizes, times, memory_usages, complexity_class)
    
    return measurements

def plot_complexity(sizes, times, memory_usages, complexity_class):
    """Generate plots for time and memory complexity."""
    plt.figure(figsize=(12, 10))
    
    # Time complexity plot
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times, 'o-', linewidth=2, markersize=8)
    plt.title(f'Time Complexity Analysis (Estimated: {complexity_class})')
    plt.xlabel('Input Size (motif length)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    
    # Memory usage plot if available
    if any(m > 0 for m in memory_usages):
        plt.subplot(2, 1, 2)
        plt.plot(sizes, memory_usages, 'o-', color='green', linewidth=2, markersize=8)
        plt.title('Memory Usage Analysis')
        plt.xlabel('Input Size (motif length)')
        plt.ylabel('Memory Usage (MB)')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Create the results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the figure in the results directory
    output_file = os.path.join(results_dir, 'complexity_analysis.png')
    plt.savefig(output_file)
    print(f"Plot saved as '{output_file}'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Analyze the algorithmic complexity of motif finder')
    parser.add_argument('midi_file', help='Path to the MIDI file')
    parser.add_argument('--start-size', type=int, default=1, help='Starting motif size')
    parser.add_argument('--end-size', type=int, default=20, help='Ending motif size')
    parser.add_argument('--step', type=int, default=1, help='Step size between motif sizes')
    parser.add_argument('--delta', type=int, default=0, help='Delta parameter for all tests')
    parser.add_argument('--gamma', type=int, default=0, help='Gamma parameter for all tests')
    parser.add_argument('--repetitions', type=int, default=3, help='Number of repetitions per measurement')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    analyze_motif_size_complexity(
        args.midi_file,
        args.start_size,
        args.end_size,
        args.step,
        args.delta,
        args.gamma,
        args.repetitions,
        not args.no_plot
    )

if __name__ == "__main__":
    main()