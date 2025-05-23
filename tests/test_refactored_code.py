#!/usr/bin/env python3
"""
Test script to verify the refactored motif finder code works correctly.

This script tests the core functionality of the motif finder with the new
modular structure and improved error handling.
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        # Test new modules
        import constants
        import exceptions
        import config
        import utils
        print("✓ New modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import new modules: {e}")
        return False
    
    try:
        # Test existing modules with new imports
        import midi_processor
        import motif_finder
        import suffix_tree
        print("✓ Existing modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import existing modules: {e}")
        return False
    
    return True

def test_configuration():
    """Test the new configuration system."""
    print("Testing configuration system...")
    
    try:
        from config import ConfigManager, MotifSearchConfig, MotifConfig
        from exceptions import ParameterError
        
        # Test valid configuration
        search_config = ConfigManager.create_search_config(delta=1, gamma=2)
        assert search_config.delta == 1
        assert search_config.gamma == 2
        print("✓ Valid search configuration created")
        
        # Test motif configuration
        motif_config = ConfigManager.create_motif_config([60, 62, 64])
        assert motif_config.pitches == [60, 62, 64]
        print("✓ Valid motif configuration created")
        
        # Test invalid configurations
        try:
            ConfigManager.create_search_config(delta=-1)
            print("✗ Should have rejected negative delta")
            return False
        except ParameterError:
            print("✓ Correctly rejected negative delta")
        
        try:
            ConfigManager.create_motif_config([])
            print("✗ Should have rejected empty motif")
            return False
        except ParameterError:
            print("✓ Correctly rejected empty motif")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic motif finding functionality."""
    print("Testing basic functionality...")
    
    try:
        from motif_finder import MotifFinder
        from exceptions import MotifFinderError
        
        # Test with a simple MIDI file
        test_file = "../data/twinkle.mid"
        if not os.path.exists(test_file):
            print("⚠ Test MIDI file not found, skipping functionality test")
            return True
        
        # Create motif finder
        finder = MotifFinder(test_file)
        print("✓ MotifFinder created successfully")
        
        # Test simple motif search
        motif = [60, 62, 64]  # C, D, E
        occurrences = finder.find_motif_occurrences(motif, delta=0, gamma=0)
        print(f"✓ Found {len(occurrences)} exact matches")
        
        # Test tolerant search
        occurrences_tolerant = finder.find_motif_occurrences(motif, delta=1, gamma=2)
        print(f"✓ Found {len(occurrences_tolerant)} tolerant matches")
        
        # Verify monotonicity (tolerant should find >= exact)
        if len(occurrences_tolerant) >= len(occurrences):
            print("✓ Monotonicity property satisfied")
        else:
            print(f"✗ Monotonicity violated: {len(occurrences_tolerant)} < {len(occurrences)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")
    
    try:
        from utils import (
            calculate_pitch_class, 
            pitch_class_to_solid_char, 
            solid_char_to_pitch_class,
            validate_motif_pitches,
            normalize_pitch_set
        )
        
        # Test pitch class calculations
        assert calculate_pitch_class(60) == 0  # C
        assert calculate_pitch_class(61) == 1  # C#
        assert calculate_pitch_class(72) == 0  # C (octave higher)
        print("✓ Pitch class calculations correct")
        
        # Test character conversions
        assert pitch_class_to_solid_char(0) == 'A'
        assert solid_char_to_pitch_class('A') == 0
        print("✓ Character conversions correct")
        
        # Test validation
        validate_motif_pitches([60, 62, 64])  # Should not raise
        print("✓ Motif validation correct")
        
        # Test pitch set normalization
        normalized = normalize_pitch_set({60, 72, 84})  # All C notes
        assert normalized == {0}
        print("✓ Pitch set normalization correct")
        
        return True
        
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running refactored code tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
        ("Utils", test_utils),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
