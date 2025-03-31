from typing import List, Tuple, Dict, Set
import os
import sys

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.midi_processor import MIDIProcessor
from src.suffix_tree import SuffixTree

class MotifFinder:
    def __init__(self, midi_file_path: str):
        """Initialize with a MIDI file path."""
        self.midi_processor = MIDIProcessor(midi_file_path)
        self.melodic_parts = self.midi_processor.extract_melodic_parts()
        self.degenerate_strings = {}
        
        # Step 1: Extract each melodic part and convert to degenerate string
        for channel, part in self.melodic_parts.items():
            self.degenerate_strings[channel] = self.midi_processor.convert_to_degenerate_string(part)
    
    def find_motif_occurrences(self, motif: List[int], delta: int, gamma: int) -> List[Tuple[int, int]]:
        """
        Find occurrences of a melodic motif with bounded mismatches using the algorithm from the paper.
        
        Args:
            motif: The melodic motif as a list of MIDI pitches
            delta: The bound on pitch mismatches (number of positions that can differ)
            gamma: The bound on Sum of Absolute Differences (SAD) between pattern and motif
            
        Returns:
            List of (channel, position) tuples indicating motif occurrences
        """
        occurrences = []
        
        # Step 2: Convert the motif M into a string P
        motif_string = self._convert_motif_to_string(motif)
        motif_length = len(motif_string)
        
        # Process each channel's degenerate string
        for channel, degenerate_string in self.degenerate_strings.items():
            # Step 3: Create a new string S = X_{$}#_{1}P#_{2}
            combined_string = SuffixTree.create_combined_string(degenerate_string, motif_string)
            
            # Count non-solid symbols in the combined string
            k = self._count_non_solid_symbols(combined_string)
            
            # Step 4: Construct the unordered suffix tree of S
            suffix_tree = SuffixTree(combined_string)
            
            # Step 5: Perform n LCE_delta queries
            n = len(degenerate_string)
            
            for i in range(n):
                # Perform LCE query with delta+k mismatch bound
                match_length, mismatches = suffix_tree.lce_query(combined_string, i, motif_length, delta + k)
                
                # Check if the mismatch is within delta, gamma bounds
                if match_length == motif_length:
                    # Verify if mismatches are within bounds
                    if self._verify_mismatches(degenerate_string[i:i+motif_length], motif_string, delta, gamma):
                        # Step 6: Return it as a motif occurrence
                        occurrences.append((channel, i))
        
        return occurrences
    
    def _convert_motif_to_string(self, motif: List[int]) -> str:
        """
        Convert a motif list to a string representation.
        Each pitch is converted to its pitch class (0-11) and represented as a character (A-L).
        """
        return ''.join(chr(ord('A') + (pitch % 12)) for pitch in motif)
    
    def _count_non_solid_symbols(self, s: str) -> int:
        """Count the number of non-solid symbols in string s (i.e., symbols representing chords)."""
        count = 0
        in_bracket = False
        
        for char in s:
            if char == '[':
                in_bracket = True
            elif char == ']':
                in_bracket = False
                count += 1  # Count each chord as one non-solid symbol
        
        return count
    
    def _verify_mismatches(self, pattern_segment: str, motif_string: str, delta: int, gamma: int) -> bool:
        """
        Verify if the mismatches between the pattern segment and motif string
        are within the delta and gamma bounds.
        
        Args:
            pattern_segment: Segment of the degenerate string
            motif_string: The motif string
            delta: The bound on pitch mismatches (number of positions that can differ)
            gamma: The bound on Sum of Absolute Differences (SAD) between pattern and motif
            
        Returns:
            True if mismatches are within bounds, False otherwise
        """
        pitch_mismatches = 0  # Count of positions where pitches differ
        sum_absolute_differences = 0  # Sum of Absolute Differences (SAD)
        
        # Compare each character
        i, j = 0, 0
        while i < len(pattern_segment) and j < len(motif_string):
            if pattern_segment[i] == '[':
                # Handle chord (non-solid symbol)
                end_bracket = pattern_segment.find(']', i)
                chord_notes = pattern_segment[i+1:end_bracket]
                
                # Find the best matching note in the chord
                best_diff = float('inf')
                motif_char = motif_string[j]
                
                for chord_char in chord_notes:
                    x_val = ord(chord_char) - ord('A')
                    y_val = ord(motif_char) - ord('A')
                    diff = min((x_val - y_val) % 12, (y_val - x_val) % 12)
                    
                    best_diff = min(best_diff, diff)
                
                if best_diff > 0:
                    pitch_mismatches += 1
                    sum_absolute_differences += best_diff
                
                i = end_bracket + 1
                j += 1
            else:
                # Handle single note (solid symbol)
                if pattern_segment[i] != motif_string[j]:
                    x_val = ord(pattern_segment[i]) - ord('A')
                    y_val = ord(motif_string[j]) - ord('A')
                    diff = min((x_val - y_val) % 12, (y_val - x_val) % 12)
                    
                    if diff > 0:
                        pitch_mismatches += 1
                        sum_absolute_differences += diff
                
                i += 1
                j += 1
        
        return pitch_mismatches <= delta and sum_absolute_differences <= gamma
