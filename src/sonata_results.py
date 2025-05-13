#!/usr/bin/env python3
"""
Run motif search on all Beethoven sonatas defined in paper_experiments.csv and summarize results.
"""
import os
import sys
import time
import gc
import ast
import pandas as pd
import mido

# Optional memory tracking
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False

# Make sure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motif_finder import MotifFinder

def main():
    # Paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, 'data', 'paper_experiments.csv')
    midi_dir = os.path.join(base, 'data', 'motif_midi')

    # Read experiments table
    df = pd.read_csv(csv_path, comment='/', skipinitialspace=True)
    results = []
    # collect per-sonata MIDI metadata
    sonata_meta = {}

    for _, row in df.iterrows():
        # Ensure the sonata identifier has two digits (e.g., '01', '02')
        sonata = str(row['SONATA']).zfill(2)
        midi_file = os.path.join(midi_dir, f"{sonata}-1.mid")
        # Parse motif list: all rotations from last column
        try:
            motif_data = ast.literal_eval(row.iloc[-1])  # last column: list of motif rotations
        except Exception as e:
            print(f"Skipped SONATA {sonata}: cannot parse motif ({e})")
            continue

        # Compute MIDI metadata once per sonata
        if sonata not in sonata_meta:
            try:
                midi = mido.MidiFile(midi_file)
                # count note-on events (velocity > 0)
                note_count = sum(1 for msg in midi if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0)
                # count distinct channels used
                channels = set(getattr(msg, 'channel', None) for msg in midi if hasattr(msg, 'channel'))
                channel_count = len(channels)
                # count chords: simultaneous note-ons at same time
                abs_time = 0
                chord_times = {}
                for msg in midi:
                    abs_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        chord_times.setdefault(abs_time, 0)
                        chord_times[abs_time] += 1
                chord_count = sum(1 for c in chord_times.values() if c > 1)
            except Exception:
                note_count = channel_count = chord_count = None
            sonata_meta[sonata] = {
                'note_count': note_count,
                'channel_count': channel_count,
                'chord_count': chord_count
            }
        # Load MIDI once per sonata for motif search
        finder = MotifFinder(midi_file)
        delta, gamma = 0, 0

        # Run search for each motif rotation
        for idx, seq in enumerate(motif_data):
            try:
                motif = [int(x) for x in seq.strip().split()]
            except Exception as e:
                print(f"Skipped SONATA {sonata} rotation {idx}: invalid motif ({e})")
                continue

            # force garbage collection before measuring baseline memory
            if has_psutil:
                gc.collect()
                mem_before = psutil.Process().memory_info().rss
            else:
                mem_before = None
            t0 = time.time()
            occs = finder.find_motif_occurrences(motif, delta, gamma)
            duration = time.time() - t0
            # force garbage collection before measuring final memory
            if has_psutil:
                gc.collect()
                mem_after = psutil.Process().memory_info().rss
            else:
                mem_after = None
            mem_used = (mem_after - mem_before) / (1024**2) if has_psutil else None

            results.append({
                'sonata': int(sonata),
                'rotation': idx,
                'midi_file': os.path.basename(midi_file),
                'motif': motif,
                'occurrences': len(occs),
                'time_s': duration,
                'memory_mb': mem_used
            })

    # Summarize
    res_df = pd.DataFrame(results)
    print(res_df)
    print(f"Total occurrences: {res_df['occurrences'].sum()}")
    print(f"Average time per sonata: {res_df['time_s'].mean():.4f}s")

    # Save summary
    out_dir = os.path.join(base, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'sonata_summary.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"Detailed summary written to {out_csv}")

    # Aggregate results per sonata (sum occurrences and time across rotations)
    agg_df = res_df.groupby('sonata').agg(
        total_occurrences=('occurrences', 'sum'),
        total_time_s=('time_s', 'sum'),
        avg_time_s=('time_s', 'mean'),
        avg_memory_mb=('memory_mb', 'mean')
    ).reset_index()
    # add MIDI metadata (notes, channels, chords)
    # sonata_meta keys are zero-padded strings, convert to int index
    meta_df = pd.DataFrame.from_dict(sonata_meta, orient='index')
    meta_df.index = meta_df.index.astype(int)
    agg_df = agg_df.merge(meta_df, left_on='sonata', right_index=True)
    agg_csv = os.path.join(out_dir, 'sonata_summary_per_sonata.csv')
    agg_df.to_csv(agg_csv, index=False)
    print(f"Aggregated per-sonata summary written to {agg_csv}")

if __name__ == '__main__':
    main()
