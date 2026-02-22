# csv-embeddings-projector

Generate embeddings from CSV files for [Google Embedding Projector](https://projector.tensorflow.org/).

## Quick Start

```bash
# 1. Setup
make venv

# 2. Test with sample data
make embed INPUT=data/sample.csv TEXT_COL=description

# 3. Upload to projector
# - Go to https://projector.tensorflow.org/
# - Click "Load" → upload output/projector_vectors.tsv
# - Click "Load" → upload output/projector_metadata.tsv
```

## Usage

```bash
# Single column
make embed INPUT=your_data.csv TEXT_COL=description

# Multiple columns combined
make embed INPUT=your_data.csv TEXT_COL="title,description"

# Custom output name
make embed INPUT=data.csv TEXT_COL=text OUTPUT=output/myproject
```

## What it does

1. Reads your CSV file
2. Generates 384-dimensional embeddings using sentence-transformers
3. Outputs two TSV files for Google's projector

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full details.
