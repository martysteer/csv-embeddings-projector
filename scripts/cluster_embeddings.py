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
from sklearn.cluster import BisectingKMeans
from tqdm import tqdm


def load_vectors(path: Path) -> np.ndarray:
    """Load a tab-separated vectors file (no header) into a numpy array."""
    return np.loadtxt(path, delimiter='\t', dtype=np.float32)


def load_metadata(path: Path) -> pd.DataFrame:
    """Load a tab-separated metadata file (with header)."""
    return pd.read_csv(path, sep='\t', dtype=str).fillna('')


def cluster(vectors: np.ndarray, levels: list[int]) -> dict[int, np.ndarray]:
    """
    Run BisectingKMeans once per level.

    BisectingKMeans recursively bisects the data — O(n log k) time and O(n)
    memory, making it practical for large corpora where AgglomerativeClustering
    and scipy Ward both require O(n²) distance matrices.
    """
    results = {}
    for n in tqdm(sorted(levels), desc="Clustering levels", unit="level"):
        model = BisectingKMeans(n_clusters=n, random_state=42)
        labels = model.fit_predict(vectors) + 1  # make 1-based for consistency
        results[n] = labels
        counts = np.bincount(labels)[1:]
        tqdm.write(f"   cluster_{n:02d}: {n} clusters "
                   f"(min={counts.min()} med={int(np.median(counts))} max={counts.max()})")
    return results


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
    parser.add_argument('--umap-dims', '-u', type=int, default=0,
                        help='Use UMAP-reduced vectors of this dimensionality '
                             'instead of raw vectors. The file '
                             'projector_umap{N}_vectors.tsv must already exist '
                             'in the same directory — run make umap first. '
                             '0 = use raw vectors (default).')
    parser.add_argument('--output', '-o',
                        help='Output path '
                             '(default: <vectors_dir>/projector_clusters_metadata.tsv)')

    args = parser.parse_args()

    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        print(f"❌ Vectors file not found: {vectors_path}")
        print("   Run 'make embed' first.")
        sys.exit(1)

    # Optionally swap in UMAP-reduced vectors for clustering
    if args.umap_dims > 0:
        umap_path = vectors_path.parent / f'projector_umap{args.umap_dims}_vectors.tsv'
        if not umap_path.exists():
            print(f"❌ UMAP vectors not found: {umap_path}")
            print(f"   Run 'make umap UMAP_DIMS={args.umap_dims}' first.")
            sys.exit(1)
        cluster_vectors_path = umap_path
    else:
        cluster_vectors_path = vectors_path

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
    print(f"  Vectors:  {cluster_vectors_path}"
          + (" (UMAP-reduced)" if args.umap_dims > 0 else ""))
    print(f"  Metadata: {metadata_path}")
    print(f"  Levels:   {args.levels}")
    print()

    vectors = load_vectors(cluster_vectors_path)
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
