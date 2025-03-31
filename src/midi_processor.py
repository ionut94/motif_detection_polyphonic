import mido
from typing import List, Tuple, Dict, Set
import numpy as np
from collections import defaultdict

class MIDIProcessor:
    def __init__(self, midi_file_path: str):
        """Initialize with the path to a MIDI file."""
        self.midi_file = mido.MidiFile(midi_file_path)
        self.midi_data = self._extract_midi_data()
    
    def _extract_midi_data(self) -> List[Tuple]:
        """Extract MIDI data as tuples (message, pitch, keyvelocity, channel, timestamp)."""
        midi_data = []
        current_time = 0
        
        for msg in self.midi_file:
            current_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                midi_data.append((
                    msg.type,
                    msg.note,
                    getattr(msg, 'velocity', 0),
                    getattr(msg, 'channel', 0),
                    current_time
                ))
        
        return midi_data
    
    def extract_melodic_parts(self) -> Dict[int, List[List[int]]]:
        """
        Extract melodic parts from the MIDI file, grouping notes into chords when they occur simultaneously.
        Returns a dictionary mapping channel to list of either individual pitches or sets of pitches (chords).
        """
        # Group notes by timestamp and channel to identify chords
        notes_by_time = defaultdict(lambda: defaultdict(list))
        
        for msg_type, pitch, velocity, channel, timestamp in self.midi_data:
            if msg_type == 'note_on' and velocity > 0:
                # Round the timestamp to group notes that are very close together
                # (within a small delta, e.g., 0.01 seconds)
                rounded_time = round(timestamp * 100) / 100
                notes_by_time[rounded_time][channel].append(pitch)
        
        # Sort timestamps to maintain the correct order
        sorted_times = sorted(notes_by_time.keys())
        
        # Create melodic parts with chord support
        melodic_parts = {}
        
        for timestamp in sorted_times:
            for channel, pitches in notes_by_time[timestamp].items():
                if channel not in melodic_parts:
                    melodic_parts[channel] = []
                
                # If there's more than one pitch at this timestamp, it's a chord
                if len(pitches) > 1:
                    melodic_parts[channel].append(pitches)
                else:
                    melodic_parts[channel].append(pitches[0])
        
        return melodic_parts
    
    def print_all_notes(self):
        """Print information about all notes in the MIDI file, including chords."""
        melodic_parts = self.extract_melodic_parts()
        total_notes = 0
        total_elements = 0
        
        for channel, part in melodic_parts.items():
            total_elements += len(part)
            for item in part:
                if isinstance(item, list):
                    total_notes += len(item)
                else:
                    total_notes += 1
        
        print(f"Total elements in the MIDI file: {total_elements} (including {total_notes} individual notes)")
        print("Notes by channel:")
        
        for channel, part in melodic_parts.items():
            chord_count = sum(1 for item in part if isinstance(item, list))
            single_notes = len(part) - chord_count
            print(f"Channel {channel}: {len(part)} elements ({single_notes} single notes, {chord_count} chords)")
            
            # Display the melodic part with chord notation
            readable_sequence = []
            for item in part:
                if isinstance(item, list):
                    readable_sequence.append(f"[{', '.join(map(str, item))}]")
                else:
                    readable_sequence.append(str(item))
            
            print(f"Sequence: {' '.join(readable_sequence)}")
            print("------------")
    
    def convert_to_degenerate_string(self, melodic_part: List) -> str:
        """
        Convert a melodic part into a degenerate string.
        In this context, we're representing each note as its pitch class (0-11).
        For chords, we convert each note in the chord to its pitch class.
        """
        degenerate_string = ""
        
        for item in melodic_part:
            if isinstance(item, list):  # It's a chord
                # Convert each pitch in the chord to its pitch class
                chord_pitch_classes = [pitch % 12 for pitch in item]
                # Sort to ensure consistent representation
                chord_pitch_classes.sort()
                # Join the chord pitch classes with a special separator
                chord_string = ''.join(chr(ord('A') + pc) for pc in chord_pitch_classes)
                degenerate_string += '[' + chord_string + ']'
            else:  # It's a single note
                # Convert the single pitch to its pitch class
                pitch_class = item % 12
                degenerate_string += chr(ord('A') + pitch_class)
        
        return degenerate_string
