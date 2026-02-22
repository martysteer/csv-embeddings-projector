# CSV to Google Embedding Projector

**Project**: csv-embeddings-projector
**Purpose**: Generate embeddings from CSV files and enrich them with hierarchical clustering, UMAP reduction, and compressed facets for visualisation in Google's Embedding Projector
**Date**: January 2026

---

## Overview

A Makefile-driven pipeline that takes a CSV, generates sentence embeddings, and produces a family of TSV files ready to load into [Google Embedding Projector](https://projector.tensorflow.org/). Each stage builds on the last and saves its output to disk, so steps can be re-run independently.

### Pipeline

```
CSV
 └─ make embed ──────────────────────── projector_vectors.tsv         (raw embeddings)
                                         projector_metadata.tsv        (all CSV columns)
      │
      ├─ make umap ───────────────────── projector_umap50_vectors.tsv  (50-dim reduction)
      │                                  projector_umap3_vectors.tsv   (3D layout)
      │
      ├─ make clusters ───────────────── projector_clusters_metadata.tsv
      │   (uses UMAP vectors by default)  └─ cluster_03, cluster_05, cluster_10, cluster_20 ...
      │
      └─ make facets ─────────────────── projector_facets_metadata.tsv
                                          └─ {col}_top10 per column
```

All output files for a given model live in `output/<model-slug>/`.

---

## Directory Structure

```
csv-embeddings-projector/
├── Makefile
├── README.md
├── data/
│   └── sample.csv
├── docs/
│   └── PROJECT_SPEC.md
├── models/                        # HuggingFace model cache
├── scripts/
│   ├── embed_csv.py               # Stage 1 — embedding
│   ├── umap_reduce.py             # Stage 2 — UMAP reduction
│   ├── cluster_embeddings.py      # Stage 3 — hierarchical clustering
│   └── compress_metadata.py       # Stage 4 — facet compression
└── output/
    └── all-minilm-l6-v2/         # One directory per model
        ├── projector_vectors.tsv
        ├── projector_metadata.tsv
        ├── projector_umap50_vectors.tsv
        ├── projector_umap3_vectors.tsv
        ├── projector_clusters_metadata.tsv
        └── projector_facets_metadata.tsv
```

---

## Models

| Variable           | Model ID                                  | Dims | Auth          |
|--------------------|-------------------------------------------|------|---------------|
| `MODEL_MINILM`     | `sentence-transformers/all-MiniLM-L6-v2` | 384  | None          |
| `MODEL_GEMMA_300M` | `google/embeddinggemma-300m`              | 768  | HF token req. |

The default model is MiniLM. Override with `MODEL=...` on any target.

EmbeddingGemma is a gated model — accept Google's licence at https://huggingface.co/google/embeddinggemma-300m then `export HF_TOKEN=hf_...` before use.

---

## Makefile Targets

### `make venv`
Creates `.venv/` and installs all dependencies. Run once on setup.

### `make embed`
Reads a CSV, generates sentence embeddings, writes two TSV files.

```bash
make embed INPUT=data/myfile.csv                      # embed all columns
make embed INPUT=data/myfile.csv TEXT_COL=description # embed one column
make embed INPUT=data/myfile.csv TEXT_COL="title,abstract"
make embed INPUT=data/myfile.csv TEXT_COL=description MODEL=$(MODEL_GEMMA_300M)
```

**Variables:** `INPUT` (required), `TEXT_COL` (default: all columns), `MODEL`, `OUTPUT`

**Outputs:**
- `projector_vectors.tsv` — one row of floats per document, no header
- `projector_metadata.tsv` — all CSV columns, with header row

### `make umap`
Reduces the raw vectors with UMAP. Two useful target dimensionalities:

```bash
make umap                    # 50-dim reduction for clustering (default)
make umap UMAP_DIMS=3        # 3D pre-computed layout for the projector
make umap UMAP_DIMS=50 UMAP_NEIGHBORS=30 UMAP_MIN_DIST=0.0
```

**Variables:** `UMAP_DIMS` (default: 50), `UMAP_NEIGHBORS` (default: 15), `UMAP_MIN_DIST` (default: 0.1)

**Output:** `projector_umap{N}_vectors.tsv`

Skips silently if the output file already exists — delete it to force re-computation.

**Dimensionality guide:**
- **50 dims** — preserves global structure while eliminating noise; feed to `make clusters`
- **3 dims** — load directly into the Embedding Projector as a pre-computed 3D layout, bypassing its own UMAP/PCA/TSNE

### `make clusters`
Performs Ward hierarchical clustering and annotates metadata with cluster labels at multiple levels of granularity.

```bash
make clusters                            # default: UMAP_DIMS=50, LEVELS=3 5 10 20
make clusters UMAP_DIMS=0               # cluster raw vectors instead of UMAP
make clusters LEVELS='5 10 25 50'       # custom cut levels
```

**Variables:** `LEVELS` (default: `3 5 10 20`), `UMAP_DIMS` (default: 50 — set to 0 to use raw vectors), `VECTORS_TSV`, `METADATA_TSV`

**Output:** `projector_clusters_metadata.tsv` — all original metadata columns plus one `cluster_NN` column per level (zero-padded so they sort correctly in the projector's colour-by dropdown).

Uses `sklearn.AgglomerativeClustering(linkage='ward')` which operates directly on feature vectors — memory stays O(n) rather than the O(n²) of scipy's pdist approach.

**Note:** requires `make umap` to have been run first when `UMAP_DIMS > 0`.

### `make facets`
The Embedding Projector only shows fields with ~15 or fewer unique values in its colour-by dropdown. This target compresses high-cardinality columns into projector-friendly equivalents.

```bash
make facets                                  # compress all columns
make facets FACET_COLS=publisher,genre       # specific columns only
make facets FACET_COLS=subject TOP_N=8       # top 8 + Other
```

**Variables:** `FACET_COLS` (default: all columns), `TOP_N` (default: 10), `METADATA_TSV`

**Output:** `projector_facets_metadata.tsv` — all original columns plus `{col}_top{N}` for each nominated column. Original columns are never modified.

Strategy: keep the N most frequent values by name; collapse everything else into a single `Other` bucket.

### `make download-model` / `make download-all-models`
Pre-download models to the local cache. Optional — `make embed` will download on first use.

### `make clean` / `make clean-all`
- `clean` — removes `output/`
- `clean-all` — removes `output/`, `models/`, `.venv/`

---

## Loading into Google Embedding Projector

Go to https://projector.tensorflow.org/ → **Load** (left panel).

You have several vector/metadata combinations to choose from:

| Vectors file | Metadata file | Use for |
|---|---|---|
| `projector_vectors.tsv` | `projector_metadata.tsv` | Basic exploration |
| `projector_umap3_vectors.tsv` | `projector_metadata.tsv` | Pre-baked 3D layout |
| `projector_vectors.tsv` | `projector_clusters_metadata.tsv` | Colour by cluster level |
| `projector_vectors.tsv` | `projector_facets_metadata.tsv` | Colour by compressed facets |

The metadata files contain all original CSV columns, so hover labels and search work regardless of which metadata file is loaded. Metadata can be swapped without re-uploading vectors.

---

## Dependencies

- Python 3.11
- `sentence-transformers` — embedding models
- `numpy`, `pandas` — data handling
- `scipy` — distance utilities
- `scikit-learn` — Ward agglomerative clustering
- `umap-learn` — UMAP dimensionality reduction
- `tqdm` — progress bars

---

## References

- [Google Embedding Projector](https://projector.tensorflow.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m)
- [Clustering and Visualising Documents using Word Embeddings](https://programminghistorian.org/en/lessons/clustering-visualizing-word-embeddings) — Programming Historian
- [UMAP documentation](https://umap-learn.readthedocs.io/)
