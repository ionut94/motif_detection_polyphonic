"""
Utility functions for the motif finding application.

This module contains common utility functions used across different modules
to reduce code duplication and improve maintainability.
"""

from typing import List, Tuple, Set

# Use try-except for relative imports
try:
    from .constants import SOLID_ALPHABET
except ImportError:
    # Fallback for standalone usage
    SOLID_ALPHABET = ''.join(chr(ord('A') + i) for i in range(12))

def calculate_pitch_class(pitch: int) -> int:
    """
    Calculate the pitch class (0-11) from a MIDI pitch.
    
    Args:
        pitch: MIDI pitch value (0-127)
        
    Returns:
        Pitch class (0-11)
    """
    return pitch % 12

def pitch_class_to_solid_char(pitch_class: int) -> str:
    """
    Convert a pitch class to its solid character representation.
    
    Args:
        pitch_class: Pitch class (0-11)
        
    Returns:
        Solid character ('A'-'L')
    """
    if not 0 <= pitch_class <= 11:
        raise ValueError(f"Invalid pitch class: {pitch_class}")
    return SOLID_ALPHABET[pitch_class]

def solid_char_to_pitch_class(char: str) -> int:
    """
    Convert a solid character to its pitch class.
    
    Args:
        char: Solid character ('A'-'L')
        
    Returns:
        Pitch class (0-11)
        
    Raises:
        ValueError: If character is not a valid solid character
    """
    if char not in SOLID_ALPHABET:
        raise ValueError(f"Invalid solid character: {char}")
    return ord(char) - ord('A')

def format_occurrence_list(occurrences: List[Tuple[int, int]]) -> str:
    """
    Format a list of occurrences for display.
    
    Args:
        occurrences: List of (channel, position) tuples
        
    Returns:
        Formatted string representation
    """
    if not occurrences:
        return "No occurrences found"
    
    result = f"Found {len(occurrences)} occurrence(s):\n"
    for channel, position in occurrences:
        result += f"  Channel {channel}, Position {position}\n"
    return result.rstrip()

def validate_motif_pitches(pitches: List[int]) -> None:
    """
    Validate that all pitches in a motif are valid MIDI pitches.
    
    Args:
        pitches: List of MIDI pitch values
        
    Raises:
        ValueError: If any pitch is invalid
    """
    for i, pitch in enumerate(pitches):
        if not isinstance(pitch, int) or not 0 <= pitch <= 127:
            raise ValueError(f"Invalid MIDI pitch at position {i}: {pitch}")

def normalize_pitch_set(pitches: Set[int]) -> Set[int]:
    """
    Normalize a set of MIDI pitches to pitch classes.
    
    Args:
        pitches: Set of MIDI pitch values
        
    Returns:
        Set of pitch classes (0-11)
    """
    return {calculate_pitch_class(pitch) for pitch in pitches}
