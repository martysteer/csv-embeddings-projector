#!/usr/bin/env python3
"""
Hierarchical clustering of pre-computed embeddings.

Reads a vectors TSV (from embed_csv.py) and the corresponding metadata TSV,
performs Ward hierarchical clustering, cuts the dendrogram at multiple levels,
and writes an augmented metadata TSV with one cluster column per level.

Usage:
  python cluster_embeddings.py output/all-minilm-l6-v2/projector_vectors.tsv \\
      --metadata output/all-minilm-l6-v2/projector_metadata.tsv \\
      --levels 3 5 10 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def load_vectors(path: Path) -> np.ndarray:
    """Load a tab-separated vectors file (no header) into a numpy array."""
    return np.loadtxt(path, delimiter='\t', dtype=np.float32)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load a tab-separated metadata file (with header)."""
    return pd.read_csv(path, sep='\t', dtype=str).fillna('')


def cluster(vectors: np.ndarray, levels: list[int]) -> dict[int, np.ndarray]:
    """
    Build a Ward linkage tree on cosine distances, then cut at each level.
    Returns a dict of {n_clusters: label_array}.

    Ward linkage requires Euclidean distance. Sentence-transformer vectors are
    L2-normalised, so Euclidean distance is monotonically equivalent to cosine
    distance — Ward works correctly here without extra normalisation.
    """
    print(f"🔗 Building linkage tree ({len(vectors)} vectors)...")
    distances = pdist(vectors, metric='euclidean')
    Z = linkage(distances, method='ward')

    results = {}
    for n in sorted(levels):
        labels = fcluster(Z, t=n, criterion='maxclust')
        results[n] = labels
        print(f"   cluster_{n:02d}: {n} clusters "
              f"(sizes: {_size_summary(labels)})")
    return results


def _size_summary(labels: np.ndarray) -> str:
    """Return a compact min/median/max size string."""
    counts = np.bincount(labels)[1:]  # fcluster labels are 1-based
    return f"min={counts.min()} med={int(np.median(counts))} max={counts.max()}"


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical clustering of pre-computed embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cluster_embeddings.py output/all-minilm-l6-v2/projector_vectors.tsv
  python cluster_embeddings.py output/all-minilm-l6-v2/projector_vectors.tsv \\
      --levels 3 5 10 20 50

Output:
  projector_clusters_metadata.tsv — original metadata + cluster_N columns
  Upload this as the metadata file in https://projector.tensorflow.org/
  then colour/filter by cluster_03, cluster_05, etc.
        """
    )
    parser.add_argument('vectors',
                        help='Path to projector_vectors.tsv')
    parser.add_argument('--metadata', '-m',
                        help='Path to projector_metadata.tsv '
                             '(default: <vectors_dir>/projector_metadata.tsv)')
    parser.add_argument('--levels', '-l', nargs='+', type=int,
                        default=[3, 5, 10, 20],
                        help='Number of clusters to cut at (default: 3 5 10 20)')
    parser.add_argument('--output', '-o',
                        help='Output path '
                             '(default: <vectors_dir>/projector_clusters_metadata.tsv)')

    args = parser.parse_args()

    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        print(f"❌ Vectors file not found: {vectors_path}")
        print("   Run 'make embed' first.")
        sys.exit(1)

    metadata_path = Path(args.metadata) if args.metadata \
        else vectors_path.parent / 'projector_metadata.tsv'
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output \
        else vectors_path.parent / 'projector_clusters_metadata.tsv'

    print("═" * 60)
    print("Hierarchical Clustering")
    print("═" * 60)
    print(f"  Vectors:  {vectors_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Levels:   {args.levels}")
    print()

    vectors = load_vectors(vectors_path)
    print(f"📐 Loaded {vectors.shape[0]} vectors × {vectors.shape[1]} dims")

    df = load_metadata(metadata_path)
    if len(df) != len(vectors):
        print(f"❌ Row count mismatch: {len(vectors)} vectors vs {len(df)} metadata rows")
        sys.exit(1)

    print()
    labels = cluster(vectors, args.levels)

    # Add a zero-padded cluster column for each level so they sort nicely
    print()
    for n, lab in sorted(labels.items()):
        col = f"cluster_{n:02d}"
        df[col] = lab.astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)

    print(f"\n✅ Written: {output_path}")
    print()
    print("Upload to https://projector.tensorflow.org/")
    print("  Use this file as the metadata — colour by cluster_03, cluster_05, etc.")


if __name__ == '__main__':
    main()
