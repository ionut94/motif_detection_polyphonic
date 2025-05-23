from typing import List, Tuple, Dict, Set, Union, Callable
import os
import sys
import numpy as np

# Use try-except for relative imports
try:
    from .midi_processor import MIDIProcessor, SOLID_ALPHABET, NON_SOLID_START_CODE_POINT
    from .suffix_tree import SuffixTree
    from .constants import SEPARATOR_1, SEPARATOR_2
    from .exceptions import MotifFinderError, SuffixTreeError, ParameterError
    from .config import MotifSearchConfig, MotifConfig
except ImportError:
    # Fallback for standalone usage
    from midi_processor import MIDIProcessor, SOLID_ALPHABET, NON_SOLID_START_CODE_POINT
    from suffix_tree import SuffixTree
    
    # Define constants
    SEPARATOR_1 = '⚑'
    SEPARATOR_2 = '⚐'
    
    # Define fallback exceptions
    class MotifFinderError(Exception):
        pass
    class SuffixTreeError(Exception):
        pass
    class ParameterError(Exception):
        pass

class MotifFinder:
    """
    A class for finding motifs in MIDI files using suffix trees and approximate matching.
    
    This class implements the algorithm described in the research paper for finding
    melodic motifs with tolerance for mismatches (delta) and pitch differences (gamma).
    """
    
    def __init__(self, midi_file_path: str):
        """
        Initialize the MotifFinder with a MIDI file.
        
        Args:
            midi_file_path: Path to the MIDI file to analyze
            
        Raises:
            FileNotFoundError: If the MIDI file doesn't exist
            MIDIProcessingError: If the MIDI file cannot be processed
        """
        if not os.path.exists(midi_file_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")
        
        try:
            self.midi_processor = MIDIProcessor(midi_file_path)
        except Exception as e:
            raise MotifFinderError(f"Failed to initialize MIDI processor: {e}")
        
        self.midi_file_path = midi_file_path

    def _calculate_pitch_diff(self, pc1: int, pc2: int) -> int:
        """Calculates the minimum difference between two pitch classes modulo 12."""
        return min((pc1 - pc2) % 12, (pc2 - pc1) % 12)

    def _create_matching_function(self, loc_map: Dict[str, Set[int]], gamma_bound: int) -> Callable[[str, str], Tuple[bool, int, int]]:
        """
        Creates the matching function M based on the algorithm's definition,
        incorporating the gamma check.

        Args:
            loc_map: Dictionary mapping non-solid chars ($d) to pitch class sets.
            gamma_bound: The maximum allowed sum of absolute differences.

        Returns:
            A function match(char1, char2) that returns:
            - bool: True if characters match according to M, False otherwise.
            - int: Number of mismatches added (0 or 1).
            - int: Pitch difference added (for gamma calculation).
        """
        solid_chars = set(SOLID_ALPHABET)
        non_solid_chars = set(loc_map.keys())
        separators = {SEPARATOR_1, SEPARATOR_2}

        memo = {} # Memoization for non-solid vs solid comparisons

        def match(char1: str, char2: str) -> Tuple[bool, int, int]:
            """
            Checks if char1 matches char2 based on the algorithm's matching table M.
            Returns: (is_match, mismatch_count_increase, gamma_increase)
            """
            # Check memoization
            if (char1, char2) in memo:
                return memo[(char1, char2)]
            if (char2, char1) in memo: # Check symmetric case for non-solid/solid
                 res_match, res_delta, res_gamma = memo[(char2, char1)]
                 # Need to return in correct order if called symmetrically
                 return res_match, res_delta, res_gamma


            # Case 1: Separators
            if char1 in separators or char2 in separators:
                is_match = (char1 == char2)
                result = (is_match, 0 if is_match else 1, 0) # Separator mismatch counts but adds 0 gamma
                # Don't memoize separator checks? Or maybe do. Let's memoize.
                memo[(char1, char2)] = result
                return result

            # Case 2: Both Solid
            if char1 in solid_chars and char2 in solid_chars:
                pc1 = ord(char1) - ord('A')
                pc2 = ord(char2) - ord('A')
                if pc1 == pc2:
                    result = (True, 0, 0)
                else:
                    diff = self._calculate_pitch_diff(pc1, pc2)
                    # Mismatch occurs if diff > 0, check if diff exceeds gamma later
                    result = (False, 1, diff)
                memo[(char1, char2)] = result
                return result

            # Case 3: Non-solid ($d) vs Solid (p)
            # Ensure char1 is non-solid, char2 is solid for consistent check
            if char1 in solid_chars and char2 in non_solid_chars:
                char1, char2 = char2, char1 # Swap them

            if char1 in non_solid_chars and char2 in solid_chars:
                non_solid_set = loc_map.get(char1)
                if non_solid_set is None: # Should not happen if loc_map is correct
                     print(f"Warning: Non-solid char '{char1}' not found in loc_map.")
                     result = (False, 1, gamma_bound + 1) # Treat as definite mismatch exceeding gamma
                     memo[(char1, char2)] = result
                     return result

                solid_pc = ord(char2) - ord('A')

                # Check if solid_pc is directly in the set (match)
                if solid_pc in non_solid_set:
                    result = (True, 0, 0)
                    memo[(char1, char2)] = result
                    return result
                else:
                    # It's a mismatch. Calculate minimum difference for gamma.
                    min_diff = gamma_bound + 1 # Initialize higher than gamma
                    if not non_solid_set: # Handle empty set case
                         min_diff = gamma_bound + 1
                    else:
                        min_diff = min(self._calculate_pitch_diff(solid_pc, chord_pc) for chord_pc in non_solid_set)

                    result = (False, 1, min_diff)
                    memo[(char1, char2)] = result
                    return result

            # Case 4: Non-solid vs Non-solid (or other unexpected cases)
            # The algorithm doesn't explicitly define $d1 vs $d2 matching.
            # It assumes comparisons are between T_S and P (solid).
            # Treat as mismatch for safety.
            is_match = (char1 == char2) # Only match if the exact same $d symbol
            result = (is_match, 0 if is_match else 1, 0) # Assume 0 gamma diff for $d vs $d
            memo[(char1, char2)] = result
            return result

        return match


    def find_motif_occurrences(self, motif_pitches: List[int], delta: int = 0, gamma: int = 0) -> List[Tuple[int, int]]:
        """
        Finds occurrences of a melodic motif in a MIDI file using the algorithm
        from the paper (solid equivalents, matching table, LCE_k).

        Args:
            motif_pitches: The melodic motif as a list of MIDI pitches.
            delta: The maximum allowed number of mismatches (default: 0).
            gamma: The maximum allowed sum of absolute pitch differences for mismatches (default: 0).

        Returns:
            A list of tuples (channel, start_index_in_T_S) for each occurrence found.
            
        Raises:
            ParameterError: If parameters are invalid
            MotifFinderError: If motif processing fails
        """
        # Validate parameters
        if delta < 0 or gamma < 0:
            raise ParameterError(f"Delta and gamma must be non-negative: delta={delta}, gamma={gamma}")
        
        if not motif_pitches:
            raise ParameterError("Motif cannot be empty")
        
        occurrences = []

        try:
            # Preprocess the motif M into its solid string P
            P = self.midi_processor.preprocess_motif(motif_pitches)
            m = len(P)
            if m == 0:
                raise MotifFinderError("Motif P is empty after preprocessing")

            # Get all channels present in the MIDI file
            channels = self.midi_processor.tilde_T_parts.keys()
            if not channels:
                raise MotifFinderError("No channels found in MIDI file")

            for channel in channels:
                try:
                    # Get the tilde_T representation for this channel
                    tilde_T = self.midi_processor.get_tilde_T(channel)
                    if not tilde_T:
                        continue # Skip empty channels

                    # Create the solid equivalent T_S and the location map loc_map
                    T_S, loc_map = self.midi_processor.create_solid_equivalent(tilde_T)
                    n = len(T_S)
                    if n < m:
                        continue # Skip if text is shorter than pattern

                    # Create the combined string S = T_S #_1 P #_2
                    S = T_S + SEPARATOR_1 + P + SEPARATOR_2
                    len_T_S = n
                    start_index_P = len_T_S + 1 # Index where P starts in S

                    # Create the matching function M incorporating gamma
                    match_function = self._create_matching_function(loc_map, gamma)

                    # Construct the Suffix Tree for S
                    suffix_tree = SuffixTree(S)

                    # Perform n-m+1 LCE_k_gamma queries using the suffix tree
                    for i in range(len_T_S - m + 1):
                        # Call the LCE query method from the SuffixTree instance
                        lce_length = suffix_tree.lce_k_gamma_query(
                            i,
                            start_index_P,
                            delta,
                            gamma,
                            match_function,
                            m  # Maximum length is the pattern length
                        )

                        # Check if the LCE length equals the motif length
                        if lce_length == m:
                            occurrences.append((channel, i))
                            
                except SuffixTreeError as e:
                    raise MotifFinderError(f"Suffix tree error for channel {channel}: {e}")
                except Exception as e:
                    # Log the error but continue with other channels
                    print(f"Warning: Error processing channel {channel}: {e}")
                    continue

            return occurrences
            
        except Exception as e:
            raise MotifFinderError(f"Failed to find motif occurrences: {e}")
    
