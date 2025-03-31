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
    
    # First, print information about all notes in the MIDI file
    midi_processor = MIDIProcessor(args.midi_file)
    print("=== MIDI File Analysis ===")
    midi_processor.print_all_notes()
    print("=========================")
    
    # Parse the motif
    motif = parse_motif(args.motif)
    print(f"Searching for motif: {motif}")
    
    # Initialize the motif finder
    finder = MotifFinder(args.midi_file)
    
    # Find motif occurrences using the algorithm from the paper
    occurrences = finder.find_motif_occurrences(motif, args.delta, args.gamma)
    
    # Print results
    if not occurrences:
        print("No motif occurrences found.")
    else:
        print(f"Found {len(occurrences)} occurrences of the motif:")
        for channel, position in occurrences:
            print(f"  Channel {channel}, Position {position}")
            
    if args.debug:
        # Print the degenerate strings for debugging
        print("\nDegenerate String Representations:")
        for channel, degenerate_string in finder.degenerate_strings.items():
            print(f"Channel {channel}: {degenerate_string}")
        
        # Print the motif string representation
        motif_string = ''.join(chr(ord('A') + (pitch % 12)) for pitch in motif)
        print(f"\nMotif String: {motif_string}")

if __name__ == "__main__":
    main()
