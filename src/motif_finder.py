from typing import List, Tuple, Dict, Set, Union, Callable # Added Union, Callable
import os
import sys
import numpy as np

# Add the src directory to the Python path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assuming project structure handles this

# Import SOLID_ALPHABET and NON_SOLID_START_CODE_POINT for character checks
from src.midi_processor import MIDIProcessor, SOLID_ALPHABET, NON_SOLID_START_CODE_POINT
from src.suffix_tree import SuffixTree # Will need modification later

# Define separators (ensure they are unique and outside other alphabets)
SEPARATOR_1 = '⚑' # Example separator 1
SEPARATOR_2 = '⚐' # Example separator 2

class MotifFinder:
    def __init__(self, midi_file_path: str):
        """Initialize with a MIDI file path."""
        self.midi_processor = MIDIProcessor(midi_file_path)
        # We don't need to store intermediate representations here anymore
        # The processor now handles tilde_T internally

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


    def find_motif_occurrences(self, motif_pitches: List[int], delta: int, gamma: int) -> List[Tuple[int, int]]:
        """
        Finds occurrences of a melodic motif in a MIDI file using the algorithm
        from the paper (solid equivalents, matching table, LCE_k).

        Args:
            motif_pitches: The melodic motif as a list of MIDI pitches.
            delta: The maximum allowed number of mismatches.
            gamma: The maximum allowed sum of absolute pitch differences for mismatches.

        Returns:
            A list of tuples (channel, start_index_in_T_S) for each occurrence found.
        """
        occurrences = []

        # Preprocess the motif M into its solid string P
        P = self.midi_processor.preprocess_motif(motif_pitches)
        m = len(P)
        if m == 0:
            print("Warning: Motif P is empty after preprocessing.")
            return []

        # Get all channels present in the MIDI file
        channels = self.midi_processor.tilde_T_parts.keys()

        for channel in channels:
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
            # Note: The gamma check is integrated here for efficiency during LCE
            match_function = self._create_matching_function(loc_map, gamma)

            # Construct the Suffix Tree for S
            # Construct the Suffix Tree for S
            try:
                # Now uses the implemented SuffixTree with Ukkonen's
                suffix_tree = SuffixTree(S)
            except Exception as e:
                 print(f"Error creating/using SuffixTree for channel {channel}: {e}")
                 continue # Skip channel if suffix tree fails


            # Perform n-m+1 LCE_k_gamma queries using the suffix tree
            for i in range(len_T_S - m + 1):
                # Call the LCE query method from the SuffixTree instance
                # It compares suffix starting at i (in T_S part of S)
                # with suffix starting at start_index_P (P part of S)
                lce_length = suffix_tree.lce_k_gamma_query(
                    i,
                    start_index_P,
                    delta,
                    gamma,
                    match_function
                )

                # Check if the LCE length equals the motif length
                if lce_length == m:
                    occurrences.append((channel, i))


        return occurrences
    
    # Removed _convert_motif_to_string (handled by MIDIProcessor.preprocess_motif)
    # Removed _count_non_solid_symbols (not needed with solid equivalents)
    # Removed _verify_mismatches (logic integrated into matching function and LCE query)
