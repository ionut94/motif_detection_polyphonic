#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
import datetime
from typing import List, Dict, Tuple, Any
import traceback

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motif_finder import MotifFinder
from src.midi_processor import MIDIProcessor
from src.test_data import get_test_cases

# Import for memory tracking
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False
    print("Note: Install 'psutil' package for memory usage tracking")

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
            result += f" | Memory: {self.memory_usage:.2f}MB"
            
        if not self.success:
            result += f"\n    Error: {self.error_message}"
            
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
        finder = MotifFinder(midi_file)
        
        # Measure execution time
        start_time = time.time()
        occurrences = finder.find_motif_occurrences(motif, delta, gamma)
        end_time = time.time()
        
        # Measure final memory usage
        final_memory = get_memory_usage()
        
        # Store results
        result.execution_time = end_time - start_time
        result.occurrences = occurrences
        result.memory_usage = final_memory - initial_memory if initial_memory > 0 else 0
        
        # Validate results if expected count is provided
        if expected_occurrences is not None:
            result.success = len(occurrences) == expected_occurrences
        else:
            result.success = True
            
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        traceback.print_exc()
    
    return result

def run_all_benchmarks(midi_folder: str, verbose: bool = False, test_filter: str = None) -> List[BenchmarkResult]:
    """Run all benchmark tests and return the results."""
    results = []
    
    # Get test cases from the test_data module
    test_cases = get_test_cases(midi_folder)
    
    # Filter test cases if filter is provided
    if test_filter:
        test_cases = [t for t in test_cases if test_filter.lower() in t["test_name"].lower()]
        if not test_cases:
            print(f"No tests found matching filter: '{test_filter}'")
            return []
    
    # Run all tests
    for i, test_case in enumerate(test_cases):
        print(f"Running test {i+1}/{len(test_cases)}: {test_case['test_name']}...")
        
        result = run_benchmark_test(
            test_case["test_name"],
            test_case["midi_file"],
            test_case["motif"],
            test_case.get("delta", 0),
            test_case.get("gamma", 0),
            test_case.get("expected_occurrences", None)
        )
        
        results.append(result)
        
        # Print result immediately if verbose
        if verbose:
            print(result)
            print("-" * 80)
    
    return results

def print_benchmark_summary(results: List[BenchmarkResult]):
    """Print a summary of benchmark results."""
    if not results:
        print("\nNo test results to summarize.")
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
        avg_memory = sum(r.memory_usage for r in results) / total_tests
        print(f"Average memory usage: {avg_memory:.2f}MB")
    
    # Print results sorted by execution time
    print("\nResults sorted by execution time (fastest to slowest):")
    for i, result in enumerate(sorted(results, key=lambda r: r.execution_time)):
        print(f"{i+1}. {result}")
    
    # Print failed tests if any
    failed_tests = [r for r in results if not r.success]
    if failed_tests:
        print("\nFailed tests:")
        for i, result in enumerate(failed_tests):
            print(f"{i+1}. {result}")

def save_results_to_file(results: List[BenchmarkResult], output_file: str):
    """Save benchmark results to a JSON file."""
    if not results:
        return
        
    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_tests": len(results),
        "passed_tests": sum(1 for r in results if r.success),
        "results": [r.to_dict() for r in results]
    }
    
    try:
        # Create the results directory if it doesn't exist
        results_dir = os.path.dirname(output_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark the motif finder implementation')
    parser.add_argument('--midi-folder', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                        help='Path to folder containing MIDI files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    parser.add_argument('--filter', type=str, help='Run only tests containing this string in their name')
    parser.add_argument('--output', '-o', type=str, help='Save results to specified JSON file')
    
    args = parser.parse_args()
    
    # If output is not specified but directory is provided, use default filename
    if args.output is None:
        # Create the results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(results_dir, f"benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    print(f"Starting benchmark using MIDI files from {args.midi_folder}")
    print("-" * 80)
    
    # Run all benchmarks
    results = run_all_benchmarks(args.midi_folder, args.verbose, args.filter)
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results to file
    save_results_to_file(results, args.output)

if __name__ == "__main__":
    main()