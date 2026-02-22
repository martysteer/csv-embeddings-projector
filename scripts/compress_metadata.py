#!/usr/bin/env python3
"""
Compress high-cardinality metadata columns into projector-friendly colour fields.

The Embedding Projector only shows columns with ~15 or fewer unique values in
its colour-by dropdown. This script adds a companion column for each nominated
column, keeping the TOP_N most frequent values by name and collapsing the rest
into 'Other'.

New columns are named {original_col}_top{N} and appended to the metadata —
original columns are never modified.

Usage:
  python compress_metadata.py output/all-minilm-l6-v2/projector_metadata.tsv \\
      --columns publisher,genre,decade \\
      --top-n 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def compress_column(series: pd.Series, top_n: int) -> pd.Series:
    """Keep the top_n most frequent values; collapse the rest to 'Other'."""
    top_values = series.value_counts().head(top_n).index
    return series.where(series.isin(top_values), other='Other')


def main():
    parser = argparse.ArgumentParser(
        description='Compress high-cardinality columns for Embedding Projector colours',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compress_metadata.py output/all-minilm-l6-v2/projector_metadata.tsv \\
      --columns publisher,subject,year
  python compress_metadata.py output/all-minilm-l6-v2/projector_metadata.tsv \\
      --columns publisher --top-n 8

Output:
  projector_facets_metadata.tsv — original columns + {col}_top{N} per column
  Upload as the metadata file in https://projector.tensorflow.org/
  Colour-by the new _top{N} columns.
        """
    )
    parser.add_argument('metadata',
                        help='Path to projector_metadata.tsv')
    parser.add_argument('--columns', '-c', default=None,
                        help='Comma-separated columns to compress (default: all columns)')
    parser.add_argument('--top-n', '-n', type=int, default=10,
                        help='Number of top values to keep (default: 10)')
    parser.add_argument('--output', '-o',
                        help='Output path (default: <metadata_dir>/projector_facets_metadata.tsv)')

    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        print("   Run 'make embed' first.")
        sys.exit(1)

    output_path = Path(args.output) if args.output \
        else metadata_path.parent / 'projector_facets_metadata.tsv'

    print("═" * 60)
    print("Compress Metadata Columns")
    print("═" * 60)
    print(f"  Metadata: {metadata_path}")
    print(f"  Top N:    {args.top_n}")
    print()

    df = pd.read_csv(metadata_path, sep='\t', dtype=str).fillna('')

    if args.columns:
        columns = [c.strip() for c in args.columns.split(',')]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"❌ Column(s) not found: {', '.join(missing)}")
            print(f"   Available: {', '.join(df.columns)}")
            sys.exit(1)
    else:
        columns = list(df.columns)
        print(f"   No columns specified — compressing all: {', '.join(columns)}")

    for col in columns:
        n_unique = df[col].nunique()
        new_col = f"{col}_top{args.top_n}"
        df[new_col] = compress_column(df[col], args.top_n)
        n_kept = df[new_col].nunique() - 1  # subtract 'Other'
        n_other = (df[new_col] == 'Other').sum()
        print(f"  {col} ({n_unique} unique values)")
        print(f"    → {new_col}: top {n_kept} named + Other ({n_other} rows)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)

    print(f"\n✅ Written: {output_path}")
    print()
    print("Upload to https://projector.tensorflow.org/")
    print(f"  Colour-by the new _top{args.top_n} columns.")


if __name__ == '__main__':
    main()
