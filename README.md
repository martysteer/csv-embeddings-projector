# csv-embeddings-projector

Generate embeddings from CSV files and explore them in [Google Embedding Projector](https://projector.tensorflow.org/) — with hierarchical clustering, UMAP reduction, and compressed facets for colour-coding.

## Quick Start

```bash
make venv
make embed INPUT=data/sample.csv TEXT_COL=description
make umap
make clusters
make facets
```

Then load the output files into https://projector.tensorflow.org/.

---

## Pipeline

Each stage saves its output to `output/<model-slug>/` and can be re-run independently.

```
make embed    →  projector_vectors.tsv + projector_metadata.tsv
make umap     →  projector_umap50_vectors.tsv  (clustering input)
              →  projector_umap3_vectors.tsv   (3D layout for projector)
make clusters →  projector_clusters_metadata.tsv  (cluster_03, cluster_05 ...)
make facets   →  projector_facets_metadata.tsv    ({col}_top10 columns)
```

---

## Targets

### `make embed`
Reads a CSV and generates sentence embeddings.

```bash
make embed INPUT=data/myfile.csv                        # all columns
make embed INPUT=data/myfile.csv TEXT_COL=description   # one column
make embed INPUT=data/myfile.csv TEXT_COL="title,abstract"
make embed INPUT=data/myfile.csv MODEL=$(MODEL_GEMMA_300M)
```

### `make umap`
Reduces vectors with UMAP. Run before `make clusters`.

```bash
make umap                      # 50-dim reduction (default, for clustering)
make umap UMAP_DIMS=3          # 3D layout — load directly into projector
make umap UMAP_NEIGHBORS=30 UMAP_MIN_DIST=0.0
```

The 3D output bypasses the projector's own UMAP/PCA/TSNE — your layout loads directly. Skips if the output file already exists; delete to recompute.

### `make clusters`
Ward hierarchical clustering at multiple levels. Requires `make umap` first (unless `UMAP_DIMS=0`).

```bash
make clusters                        # default: UMAP_DIMS=50, LEVELS=3 5 10 20
make clusters UMAP_DIMS=0            # cluster raw vectors
make clusters LEVELS='5 10 25 50'   # custom levels
```

Adds `cluster_03`, `cluster_05` etc. columns to a new metadata TSV. Colour-by these in the projector to explore the corpus at different granularities. Uses sklearn Ward — memory stays O(n), not O(n²).

### `make facets`
Compresses high-cardinality columns so they appear in the projector's colour-by dropdown (which caps out at ~15 unique values).

```bash
make facets                               # compress all columns
make facets FACET_COLS=publisher,genre    # specific columns
make facets FACET_COLS=subject TOP_N=8   # top 8 + Other
```

Keeps the top N most frequent values; everything else becomes `Other`. Adds `{col}_top10` columns — originals are never modified.

---

## Models

| Shorthand      | Model ID                                  | Dims | Auth          |
|----------------|-------------------------------------------|------|---------------|
| `MODEL_MINILM` | `sentence-transformers/all-MiniLM-L6-v2` | 384  | None          |
| `MODEL_GEMMA_300M` | `google/embeddinggemma-300m`          | 768  | HF token req. |

```bash
make download-model                              # pre-download default (MiniLM)
make download-model MODEL=$(MODEL_GEMMA_300M)   # EmbeddingGemma (gated)
make list-models
```

EmbeddingGemma requires accepting Google's licence at https://huggingface.co/google/embeddinggemma-300m then `export HF_TOKEN=hf_...`.

---

## Loading in the Projector

Go to https://projector.tensorflow.org/ → **Load**.

| Vectors | Metadata | Good for |
|---|---|---|
| `projector_vectors.tsv` | `projector_metadata.tsv` | Basic exploration |
| `projector_umap3_vectors.tsv` | `projector_metadata.tsv` | Pre-baked 3D layout |
| `projector_vectors.tsv` | `projector_clusters_metadata.tsv` | Colour by cluster level |
| `projector_vectors.tsv` | `projector_facets_metadata.tsv` | Colour by facet |

All metadata files carry the full original CSV columns, so hover labels and search work regardless. Metadata can be swapped without re-uploading vectors.

---

## All Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `INPUT` | — | Input CSV (required for embed) |
| `TEXT_COL` | all columns | Column(s) to embed, comma-separated |
| `OUTPUT` | `output/<slug>/projector` | Output file prefix |
| `UMAP_DIMS` | `50` | UMAP target dims (0 = skip UMAP) |
| `UMAP_NEIGHBORS` | `15` | Higher = more global structure |
| `UMAP_MIN_DIST` | `0.1` | Lower = tighter clusters |
| `LEVELS` | `3 5 10 20` | Cluster counts to cut dendrogram at |
| `FACET_COLS` | all columns | Columns to compress (comma-separated) |
| `TOP_N` | `10` | Named values to keep per facet column |

See `make help` for the full reference and `docs/PROJECT_SPEC.md` for architecture details.
