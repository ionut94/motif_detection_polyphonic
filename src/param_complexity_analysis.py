#!/usr/bin/env python3
"""
Parameter Complexity Analysis Tool for the Motif Finder implementation.

This script measures how execution time scales with delta and gamma parameters
to provide additional insight into the algorithmic complexity.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motif_finder import MotifFinder
from src.midi_processor import MIDIProcessor

class ParamMeasurement:
    def __init__(self, param_value: int, execution_time: float):
        self.param_value = param_value
        self.execution_time = execution_time
    
    def __str__(self) -> str:
        return f"Parameter value: {self.param_value}, Time: {self.execution_time:.6f}s"

def measure_execution_time(midi_file: str, motif: List[int], delta: int, gamma: int) -> float:
    """
    Measure execution time for finding a motif with specific parameters.
    
    Args:
        midi_file: Path to the MIDI file
        motif: The motif to search for
        delta: The delta parameter
        gamma: The gamma parameter
    
    Returns:
        Execution time in seconds
    """
    # Initialize motif finder
    finder = MotifFinder(midi_file)
    
    # Measure execution time
    start_time = time.time()
    finder.find_motif_occurrences(motif, delta, gamma)
    end_time = time.time()
    
    return end_time - start_time

def analyze_delta_complexity(midi_file: str, motif: List[int], max_delta: int = 10, 
                            gamma: int = 0, repetitions: int = 3) -> List[ParamMeasurement]:
    """
    Analyze how execution time scales with the delta parameter.
    
    Args:
        midi_file: Path to the MIDI file
        motif: Motif to search for
        max_delta: Maximum delta value to test
        gamma: Fixed gamma value for all tests
        repetitions: Number of times to repeat each measurement
        
    Returns:
        List of ParamMeasurement objects
    """
    print(f"Analyzing complexity with delta values from 0 to {max_delta}")
    print(f"Using fixed motif of length {len(motif)} and gamma={gamma}")
    print(f"Repeating each measurement {repetitions} times for accuracy")
    print("-" * 80)
    
    measurements = []
    
    for delta in range(max_delta + 1):
        print(f"Testing delta={delta}... ", end="", flush=True)
        
        # Run multiple measurements and take the average
        times = []
        
        for _ in range(repetitions):
            time_taken = measure_execution_time(midi_file, motif, delta, gamma)
            times.append(time_taken)
        
        # Take the average
        avg_time = sum(times) / len(times)
        
        measurement = ParamMeasurement(delta, avg_time)
        measurements.append(measurement)
        
        print(f"Avg time: {avg_time:.6f}s")
    
    return measurements

def analyze_gamma_complexity(midi_file: str, motif: List[int], delta: int = 0, 
                            max_gamma: int = 10, repetitions: int = 3) -> List[ParamMeasurement]:
    """
    Analyze how execution time scales with the gamma parameter.
    
    Args:
        midi_file: Path to the MIDI file
        motif: Motif to search for
        delta: Fixed delta value for all tests
        max_gamma: Maximum gamma value to test
        repetitions: Number of times to repeat each measurement
        
    Returns:
        List of ParamMeasurement objects
    """
    print(f"Analyzing complexity with gamma values from 0 to {max_gamma}")
    print(f"Using fixed motif of length {len(motif)} and delta={delta}")
    print(f"Repeating each measurement {repetitions} times for accuracy")
    print("-" * 80)
    
    measurements = []
    
    for gamma in range(max_gamma + 1):
        print(f"Testing gamma={gamma}... ", end="", flush=True)
        
        # Run multiple measurements and take the average
        times = []
        
        for _ in range(repetitions):
            time_taken = measure_execution_time(midi_file, motif, delta, gamma)
            times.append(time_taken)
        
        # Take the average
        avg_time = sum(times) / len(times)
        
        measurement = ParamMeasurement(gamma, avg_time)
        measurements.append(measurement)
        
        print(f"Avg time: {avg_time:.6f}s")
    
    return measurements

def plot_parameter_complexity(params, times, param_name):
    """Generate plots for parameter complexity analysis."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(params, times, 'o-', linewidth=2, markersize=8)
    plt.title(f'Execution Time vs {param_name.capitalize()} Parameter')
    plt.xlabel(f'{param_name.capitalize()} Value')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create the results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the figure in the results directory
    output_file = os.path.join(results_dir, f'{param_name}_complexity.png')
    plt.savefig(output_file)
    print(f"Plot saved as '{output_file}'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Analyze how complexity scales with parameters')
    parser.add_argument('midi_file', help='Path to the MIDI file')
    parser.add_argument('--motif', default='60,64,67', help='Comma-separated motif to search for')
    parser.add_argument('--analyze', choices=['delta', 'gamma', 'both'], default='both',
                       help='Which parameter to analyze')
    parser.add_argument('--max-delta', type=int, default=10, help='Maximum delta value to test')
    parser.add_argument('--max-gamma', type=int, default=10, help='Maximum gamma value to test')
    parser.add_argument('--fixed-delta', type=int, default=0, 
                       help='Fixed delta value when analyzing gamma')
    parser.add_argument('--fixed-gamma', type=int, default=0,
                       help='Fixed gamma value when analyzing delta')
    parser.add_argument('--repetitions', type=int, default=3, 
                       help='Number of repetitions per measurement')
    
    args = parser.parse_args()
    
    # Parse the motif
    try:
        motif = [int(x.strip()) for x in args.motif.split(',')]
    except ValueError:
        print("Error: Motif must be a comma-separated list of integers")
        return
    
    if args.analyze in ['delta', 'both']:
        # Analyze delta complexity
        delta_measurements = analyze_delta_complexity(
            args.midi_file,
            motif,
            args.max_delta,
            args.fixed_gamma,
            args.repetitions
        )
        
        # Plot results
        delta_values = [m.param_value for m in delta_measurements]
        delta_times = [m.execution_time for m in delta_measurements]
        plot_parameter_complexity(delta_values, delta_times, 'delta')
    
    if args.analyze in ['gamma', 'both']:
        # Analyze gamma complexity
        gamma_measurements = analyze_gamma_complexity(
            args.midi_file,
            motif,
            args.fixed_delta,
            args.max_gamma,
            args.repetitions
        )
        
        # Plot results
        gamma_values = [m.param_value for m in gamma_measurements]
        gamma_times = [m.execution_time for m in gamma_measurements]
        plot_parameter_complexity(gamma_values, gamma_times, 'gamma')

if __name__ == "__main__":
    main()