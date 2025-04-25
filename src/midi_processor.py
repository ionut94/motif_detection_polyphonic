import mido
from typing import List, Tuple, Dict, Set, Union, Any
import numpy as np
from collections import defaultdict

# Define the alphabet for solid symbols (pitch classes 0-11)
SOLID_ALPHABET = [chr(ord('A') + i) for i in range(12)]
# Define a starting point for non-solid symbol characters (outside A-L)
# Using Unicode Private Use Area characters for safety
NON_SOLID_START_CODE_POINT = 0xE000

class MIDIProcessor:
    def __init__(self, midi_file_path: str):
        """Initialize with the path to a MIDI file."""
        try:
            self.midi_file = mido.MidiFile(midi_file_path)
        except Exception as e:
            raise ValueError(f"Could not read MIDI file '{midi_file_path}': {e}")
        
        self.midi_data = self._extract_midi_data()
        # Store tilde_T for each channel
        self.tilde_T_parts: Dict[int, List[Union[int, Set[int]]]] = self._extract_tilde_T_parts()

    def _extract_midi_data(self) -> List[Tuple]:
        """Extract relevant MIDI data tuples (type, pitch, velocity, channel, absolute_time)."""
        midi_data = []
        current_time = 0.0
        for msg in self.midi_file:
            # Use cumulative time for absolute timestamp
            current_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                # Ensure required attributes exist
                pitch = getattr(msg, 'note', None)
                velocity = getattr(msg, 'velocity', 0) # note_off often lacks velocity
                channel = getattr(msg, 'channel', 0) # Default to channel 0 if missing

                if pitch is not None:
                    midi_data.append((
                        msg.type,
                        pitch,
                        velocity,
                        channel,
                        current_time # Absolute time
                    ))
        return midi_data

    def _extract_tilde_T_parts(self) -> Dict[int, List[Union[int, Set[int]]]]:
        """
        Extracts the tilde_T representation for each melodic part (channel).
        tilde_T is a list where each element is either a single pitch class (int 0-11)
        or a set of unique pitch classes for notes occurring at the same time (chords).
        """
        # Group note_on events by absolute timestamp and channel
        # Store pitch classes directly
        notes_by_time_channel = defaultdict(lambda: defaultdict(set))
        processed_times = set()

        for msg_type, pitch, velocity, channel, timestamp in self.midi_data:
            if msg_type == 'note_on' and velocity > 0:
                # Use a small tolerance for timestamp comparison to handle float inaccuracies
                # Find if a very close timestamp already exists
                existing_time = None
                for t in processed_times:
                    if abs(timestamp - t) < 1e-6: # Tolerance for float comparison
                        existing_time = t
                        break
                
                current_time_key = existing_time if existing_time is not None else timestamp
                if existing_time is None:
                     processed_times.add(timestamp)

                pitch_class = pitch % 12
                notes_by_time_channel[current_time_key][channel].add(pitch_class)

        # Sort timestamps to maintain the correct order of events
        sorted_times = sorted(notes_by_time_channel.keys())

        # Build tilde_T for each channel
        tilde_T_parts: Dict[int, List[Union[int, Set[int]]]] = defaultdict(list)
        all_channels = set(channel for _, channel_notes in notes_by_time_channel.items() for channel in channel_notes)

        for timestamp in sorted_times:
            notes_at_time = notes_by_time_channel[timestamp]
            processed_channels_at_time = set()
            for channel, pitch_class_set in notes_at_time.items():
                processed_channels_at_time.add(channel)
                if len(pitch_class_set) == 1:
                    # Solid symbol (single note)
                    tilde_T_parts[channel].append(list(pitch_class_set)[0])
                elif len(pitch_class_set) > 1:
                    # Non-solid symbol (chord) - store the set
                    tilde_T_parts[channel].append(pitch_class_set)
                # If len is 0, something went wrong, ignore for now

            # If a channel had no note at this timestamp but exists, maybe add placeholder?
            # Algorithm description doesn't specify handling rests/gaps explicitly.
            # Current approach only includes actual note events.

        return dict(tilde_T_parts) # Convert back to regular dict

    def get_tilde_T(self, channel: int) -> List[Union[int, Set[int]]]:
        """Returns the tilde_T list for a specific channel."""
        return self.tilde_T_parts.get(channel, [])

    @staticmethod
    def create_solid_equivalent(tilde_T: List[Union[int, Set[int]]]) -> Tuple[str, Dict[str, Set[int]]]:
        """
        Converts a tilde_T list into its solid equivalent string (T_S) and
        the location map (loc_map) from non-solid symbols back to their pitch class sets.

        Args:
            tilde_T: The list representation (pitch classes or sets of pitch classes).

        Returns:
            A tuple containing:
            - T_S: The solid equivalent string.
            - loc_map: A dictionary mapping non-solid characters ($d) to their original pitch class sets.
        """
        T_S_list = []
        loc_map = {}
        non_solid_counter = 0

        for item in tilde_T:
            if isinstance(item, int):
                # Solid symbol (pitch class 0-11)
                if 0 <= item < 12:
                    T_S_list.append(SOLID_ALPHABET[item])
                else:
                    # Handle unexpected pitch class values if necessary
                    # For now, maybe raise error or use a placeholder
                    print(f"Warning: Unexpected pitch class {item} encountered.")
                    T_S_list.append('?') # Placeholder for unexpected solid
            elif isinstance(item, set):
                # Non-solid symbol (chord)
                if not item: # Skip empty sets if they somehow occur
                    continue
                # Generate the unique non-solid character '$d'
                dollar_char = chr(NON_SOLID_START_CODE_POINT + non_solid_counter)
                T_S_list.append(dollar_char)
                # Store the mapping: $d -> {pitch classes}
                loc_map[dollar_char] = item
                non_solid_counter += 1
            else:
                 # Handle unexpected data types if necessary
                 print(f"Warning: Unexpected item type {type(item)} in tilde_T.")
                 T_S_list.append('?') # Placeholder for unexpected type


        T_S = "".join(T_S_list)
        return T_S, loc_map

    @staticmethod
    def preprocess_motif(motif_pitches: List[int]) -> str:
        """
        Preprocesses a motif (list of MIDI pitches) into its solid string P.
        Motifs are assumed to be solid (no chords).
        """
        # Motif P is assumed to be solid according to the general problem description
        # Convert each pitch to its pitch class (0-11) and then to the corresponding character (A-L)
        P_list = []
        for pitch in motif_pitches:
             pitch_class = pitch % 12
             if 0 <= pitch_class < 12:
                 P_list.append(SOLID_ALPHABET[pitch_class])
             else:
                 print(f"Warning: Unexpected pitch class {pitch_class} in motif.")
                 P_list.append('?')
        return "".join(P_list)

    def print_all_notes(self):
        """Prints information about the extracted tilde_T parts."""
        total_elements = 0
        total_notes_in_chords = 0
        total_single_notes = 0

        for channel, part in self.tilde_T_parts.items():
            total_elements += len(part)
            for item in part:
                if isinstance(item, set):
                    total_notes_in_chords += len(item)
                elif isinstance(item, int):
                    total_single_notes += 1

        total_individual_notes = total_single_notes + total_notes_in_chords
        print(f"Total elements in tilde_T representation: {total_elements} (including {total_individual_notes} individual pitch classes)")
        print("tilde_T by channel:")

        for channel, part in self.tilde_T_parts.items():
            chord_count = sum(1 for item in part if isinstance(item, set))
            single_note_count = sum(1 for item in part if isinstance(item, int))
            print(f"Channel {channel}: {len(part)} elements ({single_note_count} single notes, {chord_count} chords)")

            # Display the tilde_T sequence with inline position markers every 10 notes
            annotated_sequence = []
            for idx, item in enumerate(part):
                if idx % 10 == 0:
                    annotated_sequence.append(f"[{idx}]")
                if isinstance(item, set):
                    sorted_pcs = sorted(list(item))
                    annotated_sequence.append(f"{{{','.join(map(str, sorted_pcs))}}}")
                elif isinstance(item, int):
                    annotated_sequence.append(str(item))
                else:
                    annotated_sequence.append("?")  # Should not happen

            print(f"Sequence: {' '.join(annotated_sequence)}")
            # Also display the solid-string representation (T_S) and mapping for this channel
            T_S, loc_map = self.create_solid_equivalent(part)
            # Print the T_S string with inline position markers every 10 symbols
            # print(f"Solid string (T_S)): {T_S}")
            annotated_T_S = []
            for idx, ch in enumerate(T_S):
                if idx % 10 == 0:
                    annotated_T_S.append(f"[{idx}]")
                annotated_T_S.append(ch)
            print(f"Annotated T_S: {' '.join(annotated_T_S)}")
            # Print loc_map for non-solid symbols
            loc_map_str = ", ".join(f"'{k}':{{{','.join(map(str, sorted(v)))}}}" for k, v in loc_map.items())
            print(f"loc_map: {{{loc_map_str}}}")
            print("------------")
