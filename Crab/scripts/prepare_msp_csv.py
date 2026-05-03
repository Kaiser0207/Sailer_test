#!/usr/bin/env python3
"""
prepare_msp_csv.py

Converts the MSP-Podcast 2.0 `labels_consensus.csv` into the format
required by Crab's training/testing scripts:
  - Filters out EmoClass 'O' (Other) and 'X' (No agreement)
  - Converts EmoClass single-letter to 8 one-hot columns
  - Merges transcript text into a 'Text' column
  - Outputs the processed CSV

Usage:
    python prepare_msp_csv.py \
        --labels_csv /path/to/labels_consensus.csv \
        --transcripts_dir /path/to/Transcripts/ \
        --output_csv /path/to/msp2_processed_labels.csv
"""

import os
import argparse
import csv
from collections import Counter

# EmoClass single-letter to full name mapping (Crab's 8 classes)
EMO_MAP = {
    'A': 'Angry',
    'S': 'Sad',
    'H': 'Happy',
    'U': 'Surprise',
    'F': 'Fear',
    'D': 'Disgust',
    'C': 'Contempt',
    'N': 'Neutral',
}

EMOTION_COLUMNS = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']


def main():
    parser = argparse.ArgumentParser(description="Prepare MSP-Podcast CSV for Crab training")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to labels_consensus.csv")
    parser.add_argument("--transcripts_dir", type=str, required=True,
                        help="Path to Transcripts/ directory")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output path for processed CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("Crab MSP-Podcast Data Preparation")
    print("=" * 60)
    print(f"  Labels CSV:     {args.labels_csv}")
    print(f"  Transcripts:    {args.transcripts_dir}")
    print(f"  Output CSV:     {args.output_csv}")
    print()

    # --- Read input CSV ---
    rows = []
    with open(args.labels_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_raw = len(rows)
    print(f"Total raw samples: {total_raw}")

    # --- Filter out O (Other) and X (No agreement) ---
    filtered_rows = [r for r in rows if r['EmoClass'] in EMO_MAP]
    filtered_out = total_raw - len(filtered_rows)
    print(f"Filtered out {filtered_out} samples (EmoClass O/X)")
    print(f"Remaining samples: {len(filtered_rows)}")

    # --- Process each row ---
    output_rows = []
    text_found = 0
    text_missing = 0
    split_counter = Counter()
    emo_counter = Counter()

    for row in filtered_rows:
        filename = row['FileName']
        emo_class = row['EmoClass']
        emo_name = EMO_MAP[emo_class]

        # Build one-hot encoding
        one_hot = {emo: 1 if emo == emo_name else 0 for emo in EMOTION_COLUMNS}

        # Read transcript
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(args.transcripts_dir, f"{base_name}.txt")

        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as tf:
                text = tf.read().strip()
            text_found += 1
        else:
            text = ""
            text_missing += 1

        # Build output row
        out_row = {
            'FileName': filename,
        }
        out_row.update(one_hot)
        out_row['Text'] = text
        out_row['EmoAct'] = row['EmoAct']
        out_row['EmoVal'] = row['EmoVal']
        out_row['EmoDom'] = row['EmoDom']
        out_row['SpkrID'] = row['SpkrID']
        out_row['Gender'] = row['Gender']
        out_row['Split_Set'] = row['Split_Set']

        output_rows.append(out_row)
        split_counter[row['Split_Set']] += 1
        emo_counter[emo_name] += 1

    # --- Write output CSV ---
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    fieldnames = ['FileName'] + EMOTION_COLUMNS + ['Text', 'EmoAct', 'EmoVal', 'EmoDom', 'SpkrID', 'Gender', 'Split_Set']

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # --- Print statistics ---
    print()
    print("=" * 60)
    print("Output Statistics")
    print("=" * 60)
    print(f"  Total output samples: {len(output_rows)}")
    print(f"  Text found: {text_found} ({100*text_found/len(output_rows):.1f}%)")
    print(f"  Text missing: {text_missing} ({100*text_missing/len(output_rows):.1f}%)")
    print()

    print("Split distribution:")
    for split in ['Train', 'Development', 'Test1', 'Test2']:
        count = split_counter.get(split, 0)
        print(f"  {split:15s}: {count:>7d}")
    print()

    print("Emotion distribution:")
    for emo in EMOTION_COLUMNS:
        count = emo_counter.get(emo, 0)
        pct = 100 * count / len(output_rows)
        print(f"  {emo:10s}: {count:>7d} ({pct:5.1f}%)")

    print()
    print(f"CSV saved to: {args.output_csv}")
    print("Done!")


if __name__ == "__main__":
    main()
