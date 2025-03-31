#!/usr/bin/env python3
"""
This file contains test data for the benchmark suite.
Each test case includes a name, the MIDI file to test with,
the motif to search for, and parameters like delta and gamma.
"""

import os

def get_test_cases(midi_folder):
    """
    Returns a list of test cases for the benchmark suite.
    
    Each test case is a dictionary with these keys:
    - test_name: String name of the test
    - midi_file: Path to the MIDI file
    - motif: List of MIDI pitches for the motif
    - delta: Maximum allowed pitch mismatches (optional, default 0)
    - gamma: Maximum allowed Sum of Absolute Differences (optional, default 0)
    - expected_occurrences: Expected number of occurrences (optional)
    """
    return [
        # Basic tests on example1chords.mid
        {
            "test_name": "C major triad exact match",
            "midi_file": os.path.join(midi_folder, "example1chords.mid"),
            "motif": [60, 64, 67],  # C major triad (C E G)
            "delta": 0,
            "gamma": 0,
            "expected_occurrences": None  # Will be considered successful if it runs without errors
        },
        {
            "test_name": "C major triad with 1 pitch mismatch",
            "midi_file": os.path.join(midi_folder, "example1chords.mid"),
            "motif": [60, 64, 67],
            "delta": 1,
            "gamma": 2,
            "expected_occurrences": None
        },
        {
            "test_name": "C major scale fragment",
            "midi_file": os.path.join(midi_folder, "example1chords.mid"),
            "motif": [60, 62, 64, 65, 67],  # C D E F G
            "delta": 1,
            "gamma": 3,
            "expected_occurrences": None
        },
        
        # Tests on twinkle.mid
        {
            "test_name": "Twinkle twinkle opening motif exact",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [60, 60, 67, 67, 69, 69, 67],
            "delta": 0,
            "gamma": 0,
            "expected_occurrences": None
        },
        {
            "test_name": "Twinkle twinkle opening motif with allowed mismatches",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [60, 60, 67, 67, 69, 69, 67],
            "delta": 2,
            "gamma": 4,
            "expected_occurrences": None
        },
        {
            "test_name": "Second phrase of Twinkle",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [65, 65, 64, 64, 62, 62, 60],
            "delta": 1,
            "gamma": 2,
            "expected_occurrences": None
        },
        
        # Performance tests with larger motifs
        {
            "test_name": "Performance test with larger motif",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60],
            "delta": 3,
            "gamma": 5,
            "expected_occurrences": None
        },
        
        # Edge cases
        {
            "test_name": "Single note motif",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [60],
            "delta": 0,
            "gamma": 0,
            "expected_occurrences": None
        },
        {
            "test_name": "Non-existent motif",
            "midi_file": os.path.join(midi_folder, "example1chords.mid"),
            "motif": [61, 63, 66],  # Motif that shouldn't exist in the file
            "delta": 0,
            "gamma": 0,
            "expected_occurrences": 0
        },
        {
            "test_name": "Large delta and gamma values",
            "midi_file": os.path.join(midi_folder, "twinkle.mid"),
            "motif": [60, 62, 64, 65, 67],
            "delta": 5,
            "gamma": 10,
            "expected_occurrences": None
        }
    ]