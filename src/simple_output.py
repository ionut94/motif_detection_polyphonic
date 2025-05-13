import mido
import sys

def print_midi_info(filepath):
    """
    Prints all available information about a MIDI file.
    """
    try:
        mid = mido.MidiFile(filepath)
        print(f"Successfully opened {filepath}")
        print(f"Type: {mid.type}")
        print(f"Length: {mid.length} seconds")
        print(f"Ticks per beat: {mid.ticks_per_beat}")
        if hasattr(mid, 'charset'):
            print(f"Charset: {mid.charset}")

        print(f"Number of tracks: {len(mid.tracks)}")
        
        # Collect all unique channels with note_on events
        active_channels = set()
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'channel'):
                    active_channels.add(msg.channel)
        
        print(f"Active MIDI channels with note_on events: {sorted(list(active_channels))}")

        for i, track in enumerate(mid.tracks):
            print(f"--- Track {i}: {track.name} ---")
            print(f"  Number of messages: {len(track)}")
            
            msg_summary = {}
            note_on_channels_in_track = set() # New: To store channels for note_on messages in this track
            for msg in track:
                msg_summary[msg.type] = msg_summary.get(msg.type, 0) + 1
                if msg.type == 'note_on' and hasattr(msg, 'channel'): # New: Check channel for note_on
                    note_on_channels_in_track.add(msg.channel)
            
            if msg_summary:
                print("  Message types:")
                for msg_type, count in msg_summary.items():
                    print(f"    {msg_type}: {count}")
                if note_on_channels_in_track: # New: Print channels if note_on messages exist
                    print(f"    note_on_channels: {sorted(list(note_on_channels_in_track))}")
            else:
                print("  No messages in this track.")
        if mid.filename:
            print(f"Original filename: {mid.filename}")


    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_output.py <path_to_midi_file>")
        sys.exit(1)
    
    midi_file_path = sys.argv[1]
    print_midi_info(midi_file_path)

