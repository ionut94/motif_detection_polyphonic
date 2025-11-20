import argparse
import os
import sys
import traceback
from typing import List

# Use try-except for relative imports
try:
    from .motif_finder import MotifFinder
    from .midi_processor import MIDIProcessor
    from .config import ConfigManager, MotifSearchConfig
    from .exceptions import MotifFinderError, ParameterError
except ImportError:
    # Fallback for standalone usage
    from motif_finder import MotifFinder
    from midi_processor import MIDIProcessor
    # Define fallback exceptions
    class MotifFinderError(Exception):
        pass
    class ParameterError(Exception):
        pass

def parse_motif(motif_str: str) -> List[int]:
    """Parse a comma-separated string of MIDI pitches into a list of integers.
    
    Args:
        motif_str: Comma-separated string of MIDI pitch values
        
    Returns:
        List of integer MIDI pitch values
        
    Raises:
        ParameterError: If the string cannot be parsed as valid MIDI pitches
    """
    try:
        return ConfigManager.parse_motif_string(motif_str)
    except NameError:
        # Fallback implementation
        try:
            pitches = [int(x.strip()) for x in motif_str.split(',')]
            for pitch in pitches:
                if not 0 <= pitch <= 127:
                    raise ParameterError(f"Invalid MIDI pitch: {pitch}")
            return pitches
        except ValueError as e:
            raise ParameterError(f"Invalid motif string: {e}")

def main():
    """Main function to parse arguments and run motif finding."""
    parser = argparse.ArgumentParser(description='Find melodic motifs in MIDI files using the algorithm from the paper')
    parser.add_argument('midi_file', help='Path to the MIDI file')
    parser.add_argument('motif', help='Comma-separated list of MIDI pitches representing the motif')
    parser.add_argument('--delta', type=int, default=0, help='Maximum allowed per-note pitch-class difference (in semitones)')
    parser.add_argument('--gamma', type=int, default=0, help='Maximum allowed Sum of Absolute Differences (SAD) between pattern and motif')
    parser.add_argument('--debug', action='store_true', help='Print debug information about the MIDI file')

    args = parser.parse_args()

    # Initialize MIDI processor and Motif Finder
    try:
        midi_processor = MIDIProcessor(args.midi_file)
        finder = MotifFinder(args.midi_file)
    except MotifFinderError as e:
        print(f"Error initializing motif finder: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during initialization: {e}")
        sys.exit(1)

    # Print MIDI analysis if requested or by default
    print("=== MIDI File Analysis ===")
    midi_processor.print_all_notes()
    print("=========================")

    # Parse and validate the motif using the new configuration system
    try:
        motif = parse_motif(args.motif)
        # Create configuration objects for validation if available
        try:
            search_config = ConfigManager.create_search_config(
                delta=args.delta, 
                gamma=args.gamma
            )
            motif_config = ConfigManager.create_motif_config(motif)
        except NameError:
            # Fallback validation
            if args.delta < 0:
                raise ParameterError(f"Delta must be non-negative: {args.delta}")
            if args.gamma < 0:
                raise ParameterError(f"Gamma must be non-negative: {args.gamma}")
    except ParameterError as e:
        print(f"Parameter error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing motif: {e}")
        sys.exit(1)

    # Compute normalized (solid string) motif
    motif_string = midi_processor.preprocess_motif(motif)
    # Print both original and normalized motif
    print(f"Searching for motif: {motif} (solid string: {motif_string}) with delta={args.delta}, gamma={args.gamma}")

    # Find motif occurrences using the algorithm from the paper
    try:
        occurrences = finder.find_motif_occurrences(motif, args.delta, args.gamma)
    except MotifFinderError as e:
        print(f"Motif finder error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during motif search: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Print results
    if not occurrences:
        print("No motif occurrences found.")
    else:
        print(f"Found {len(occurrences)} occurrences of the motif:")
        for channel, position in occurrences:
            print(f"  Channel {channel}, Position {position}")

    if args.debug:
        # Print T_S and loc_map for debugging
        print("\n--- Solid Strings (T_S) and Location Maps (loc_map) ---")
        for channel, tilde_T in midi_processor.tilde_T_parts.items():
            if not tilde_T: continue
            T_S, loc_map = midi_processor.create_solid_equivalent(tilde_T)
            print(f"\nChannel {channel}:")
            print(f"  T_S ({len(T_S)} chars): {T_S}")
            loc_map_str = ", ".join([f"'{k}':{{{','.join(map(str, sorted(v)))}}}" for k, v in loc_map.items()])
            print(f"  loc_map: {{{loc_map_str}}}")

        # Print the motif solid string representation
        motif_string = midi_processor.preprocess_motif(motif) # Use processor method
        print(f"\nMotif Solid String (P): {motif_string}")

if __name__ == "__main__":
    main()
