import argparse
import os
import sys
from typing import List

# Add the src directory to the Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motif_finder import MotifFinder
from src.midi_processor import MIDIProcessor

def parse_motif(motif_str: str) -> List[int]:
    """Parse a comma-separated string of MIDI pitches into a list of integers."""
    try:
        return [int(x.strip()) for x in motif_str.split(',')]
    except ValueError:
        raise ValueError("Motif must be a comma-separated list of MIDI pitch values")

def main():
    parser = argparse.ArgumentParser(description='Find melodic motifs in MIDI files using the algorithm from the paper')
    parser.add_argument('midi_file', help='Path to the MIDI file')
    parser.add_argument('motif', help='Comma-separated list of MIDI pitches representing the motif')
    parser.add_argument('--delta', type=int, default=0, help='Maximum allowed pitch mismatches (number of positions that can differ)')
    parser.add_argument('--gamma', type=int, default=0, help='Maximum allowed Sum of Absolute Differences (SAD) between pattern and motif')
    parser.add_argument('--debug', action='store_true', help='Print debug information about the MIDI file')

    args = parser.parse_args()

    # Initialize MIDI processor and Motif Finder
    try:
        midi_processor = MIDIProcessor(args.midi_file)
        finder = MotifFinder(args.midi_file) # MotifFinder uses the same processor instance implicitly now
    except Exception as e:
        print(f"Error initializing processor or finder: {e}")
        sys.exit(1)

    # Print MIDI analysis if requested or by default
    print("=== MIDI File Analysis ===")
    midi_processor.print_all_notes()
    print("=========================")

    # Parse the motif
    try:
        motif = parse_motif(args.motif)
    except ValueError as e:
        print(f"Error parsing motif: {e}")
        sys.exit(1)

    # Compute normalized (solid string) motif
    motif_string = midi_processor.preprocess_motif(motif)
    # Print both original and normalized motif
    print(f"Searching for motif: {motif} (solid string: {motif_string}) with delta={args.delta}, gamma={args.gamma}")

    # Find motif occurrences using the algorithm from the paper
    try:
        occurrences = finder.find_motif_occurrences(motif, args.delta, args.gamma)
    except Exception as e:
        print(f"Error during motif search: {e}")
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
