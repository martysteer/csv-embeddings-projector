#!/usr/bin/env python3
"""
Reduce high-dimensional embeddings with UMAP.

Reads a projector_vectors.tsv and writes a reduced version alongside it.
The output is a valid Embedding Projector vectors file at any dimensionality:
  - High dims (e.g. 50): use as clustering input to avoid O(n²) RAM in Ward
  - Low dims (3): load directly into the projector as a pre-computed layout,
    bypassing its own UMAP/PCA/TSNE reduction entirely

Usage:
  python umap_reduce.py output/all-minilm-l6-v2/projector_vectors.tsv
  python umap_reduce.py output/all-minilm-l6-v2/projector_vectors.tsv --dims 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_vectors(path: Path) -> np.ndarray:
    """Load a tab-separated vectors file (no header) into a numpy array."""
    print(f"📐 Loading vectors: {path}")
    vectors = np.loadtxt(path, delimiter='\t', dtype=np.float32)
    print(f"   Shape: {vectors.shape[0]} × {vectors.shape[1]}")
    return vectors


def reduce(vectors: np.ndarray, dims: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    """Run UMAP reduction. Imports lazily so startup is fast on --help."""
    import umap
    from tqdm import tqdm

    print(f"\n🗺️  Running UMAP: {vectors.shape[1]} → {dims} dims")
    print(f"   n_neighbors={n_neighbors}  min_dist={min_dist}")
    print(f"   (this may take a few minutes for large corpora)\n")

    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        low_memory=True,
        verbose=True,
    )
    return reducer.fit_transform(vectors).astype(np.float32)


def save_vectors(vectors: np.ndarray, path: Path) -> None:
    """Save as tab-separated floats with no header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in vectors:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
    print(f"\n✅ Written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reduce embeddings with UMAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dimensionality guide:
  --dims 50   Clustering input — preserves structure, eliminates noise.
              Feed to 'make clusters UMAP_DIMS=50' instead of raw vectors.
  --dims 3    Pre-computed 3D layout for the Embedding Projector.
              Load as vectors — the projector uses your layout directly,
              bypassing its own UMAP/PCA/TSNE.

Examples:
  python umap_reduce.py output/all-minilm-l6-v2/projector_vectors.tsv
  python umap_reduce.py output/all-minilm-l6-v2/projector_vectors.tsv --dims 3
  python umap_reduce.py output/all-minilm-l6-v2/projector_vectors.tsv \\
      --dims 50 --n-neighbors 30 --min-dist 0.1
        """
    )
    parser.add_argument('vectors',
                        help='Path to projector_vectors.tsv')
    parser.add_argument('--dims', '-d', type=int, default=50,
                        help='Target dimensionality (default: 50)')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors — higher = more global structure '
                             '(default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist — lower = tighter clusters '
                             '(default: 0.1)')
    parser.add_argument('--output', '-o',
                        help='Output path (default: <vectors_dir>/projector_umap{dims}_vectors.tsv)')

    args = parser.parse_args()

    vectors_path = Path(args.vectors)
    if not vectors_path.exists():
        print(f"❌ Vectors file not found: {vectors_path}")
        print("   Run 'make embed' first.")
        sys.exit(1)

    output_path = Path(args.output) if args.output \
        else vectors_path.parent / f'projector_umap{args.dims}_vectors.tsv'

    if output_path.exists():
        print(f"✓ Already exists, skipping: {output_path}")
        print("  Delete the file to force re-computation.")
        sys.exit(0)

    print("═" * 60)
    print("UMAP Dimensionality Reduction")
    print("═" * 60)

    vectors = load_vectors(vectors_path)

    if args.dims >= vectors.shape[1]:
        print(f"❌ --dims {args.dims} must be less than input dims ({vectors.shape[1]})")
        sys.exit(1)

    reduced = reduce(vectors, args.dims, args.n_neighbors, args.min_dist)
    print(f"   Output shape: {reduced.shape[0]} × {reduced.shape[1]}")

    save_vectors(reduced, output_path)

    print()
    if args.dims <= 3:
        print("💡 3D layout tip: load this as the vectors file in the Embedding")
        print("   Projector — it will use your pre-computed layout directly.")
    else:
        print(f"💡 Clustering tip: run 'make clusters UMAP_DIMS={args.dims}'")
        print("   to cluster these reduced vectors instead of the raw embeddings.")


if __name__ == '__main__':
    main()
