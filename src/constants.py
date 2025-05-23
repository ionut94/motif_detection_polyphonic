"""
Constants used throughout the motif finding application.

This module centralizes all constants to improve maintainability and reduce
code duplication across the project.
"""

# Separators used in string concatenation for suffix tree construction
SEPARATOR_1 = '⚑'  # Primary separator between text and pattern
SEPARATOR_2 = '⚐'  # Secondary separator (if needed)

# Terminal character for suffix tree construction
TERMINAL_CHAR = chr(0)

# MIDI processing constants
NON_SOLID_START_CODE_POINT = 0x1F300  # Unicode starting point for non-solid characters

# Solid alphabet for MIDI note representation
SOLID_ALPHABET = ''.join(chr(ord('A') + i) for i in range(12))  # A-L representing pitch classes 0-11

# Error messages
class ErrorMessages:
    """Centralized error messages for consistent error handling."""
    
    MIDI_FILE_NOT_FOUND = "MIDI file not found: {}"
    INVALID_MOTIF_FORMAT = "Motif must be a comma-separated list of MIDI pitch values"
    SUFFIX_TREE_BUILD_FAILED = "Failed to build suffix tree: {}"
    LCE_QUERY_FAILED = "LCE query failed: {}"
    INVALID_PARAMETERS = "Invalid parameters: delta={}, gamma={}"
    EMPTY_MIDI_FILE = "MIDI file contains no note events"

# Configuration defaults
class Config:
    """Default configuration values."""
    
    DEFAULT_DELTA = 0
    DEFAULT_GAMMA = 0
    MIN_MOTIF_LENGTH = 2
    MAX_MOTIF_LENGTH = 100
    DEFAULT_OUTPUT_FORMAT = 'csv'
    
    # Memory and performance limits
    MAX_STRING_LENGTH = 1_000_000  # Maximum combined string length for suffix tree
    MAX_CHANNELS = 16  # Standard MIDI channel limit
