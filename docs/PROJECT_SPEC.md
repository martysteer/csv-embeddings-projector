# CSV to Google Embeddings Projector

**Project**: csv-projector  
**Purpose**: Generate embeddings from CSV files for visualisation in Google's Embedding Projector  
**Date**: January 2026

---

## Overview

A simple command-line tool that:
1. Reads a CSV file
2. Generates embeddings for specified text column(s)
3. Outputs TSV files compatible with [Google Embedding Projector](https://projector.tensorflow.org/)

### Output Files

The tool generates two TSV files:

| File | Description |
|------|-------------|
| `{output}_vectors.tsv` | Tab-separated embedding vectors (no header) |
| `{output}_metadata.tsv` | Tab-separated metadata with column headers |

---

## Architecture

```
input.csv → embed_csv.py → output_vectors.tsv
                         → output_metadata.tsv
                         
Upload both to https://projector.tensorflow.org/
```

### Directory Structure

```
csv-projector/
├── Makefile                 # Build automation
├── PROJECT_SPEC.md          # This file
├── .venv/                   # Python virtual environment
├── models/                  # Cached sentence-transformers model
├── scripts/
│   └── embed_csv.py         # Main embedding script
└── output/                  # Generated TSV files
```

---

## Configuration

### Embedding Model

Default: `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensions
- Fast, good quality
- ~90MB download

Alternative models can be specified via `--model` flag.

### Input Requirements

- CSV file with header row
- At least one text column for embedding
- UTF-8 encoding recommended

---

## Usage

```bash
# Setup virtual environment
make venv

# Embed a CSV file (embed the 'description' column)
make embed INPUT=data/myfile.csv TEXT_COL=description

# Or with multiple columns combined
make embed INPUT=data/myfile.csv TEXT_COL="title,description"

# Or use the script directly
.venv/bin/python scripts/embed_csv.py data/myfile.csv \
    --text-columns description \
    --output output/myfile
```

### Output

After running, upload to Google Embedding Projector:
1. Go to https://projector.tensorflow.org/
2. Click "Load" in the left panel
3. Upload `output_vectors.tsv` as vectors
4. Upload `output_metadata.tsv` as metadata

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make venv` | Create Python virtual environment |
| `make embed INPUT=... TEXT_COL=...` | Generate embeddings |
| `make clean` | Remove output files |
| `make clean-all` | Remove venv and all generated files |
| `make info` | Show project status |

---

## Dependencies

- Python 3.9+
- sentence-transformers
- numpy
- pandas

---

## References

- [Google Embedding Projector](https://projector.tensorflow.org/)
- [Sentence Transformers](https://www.sbert.net/)
- Adapted from: `embeddings-projector-converor` (JSON→TSV converter)
