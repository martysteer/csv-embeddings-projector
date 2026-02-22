# csv-embeddings-projector

Generate embeddings from CSV files for [Google Embedding Projector](https://projector.tensorflow.org/).

## Quick Start

```bash
# 1. Setup
make venv

# 2. Test with sample data (uses MiniLM by default)
make embed INPUT=data/sample.csv TEXT_COL=description

# 3. Upload to projector
# - Go to https://projector.tensorflow.org/
# - Click "Load" → upload output/all-minilm-l6-v2/projector_vectors.tsv
# - Click "Load" → upload output/all-minilm-l6-v2/projector_metadata.tsv
```

## Models

| Shorthand   | Model ID                              | Dims | Auth required |
|-------------|---------------------------------------|------|---------------|
| `minilm`    | `sentence-transformers/all-MiniLM-L6-v2` | 384  | No            |
| `gemma-300m`| `google/embeddinggemma-300m`          | 768  | Yes (HF token)|

```bash
# See all models
make list-models

# Pre-download a model (optional — embed will download on first run)
make download-model                              # default: MiniLM
make download-model MODEL=$(MODEL_GEMMA_300M)   # EmbeddingGemma
```

Output files are written to `output/<model-slug>/` so runs with different
models never overwrite each other.

### EmbeddingGemma (gated model)

`google/embeddinggemma-300m` requires a Hugging Face account and acceptance
of Google's licence at https://huggingface.co/google/embeddinggemma-300m.

Once accepted, create a token at https://huggingface.co/settings/tokens and
export it before running:

```bash
export HF_TOKEN=hf_...
make download-model MODEL=$(MODEL_GEMMA_300M)
make embed INPUT=data/sample.csv TEXT_COL=description MODEL=$(MODEL_GEMMA_300M)
```

## Usage

```bash
# Single column (default model: MiniLM)
make embed INPUT=your_data.csv TEXT_COL=description

# Multiple columns combined
make embed INPUT=your_data.csv TEXT_COL="title,description"

# Choose a model
make embed INPUT=data.csv TEXT_COL=text MODEL=$(MODEL_GEMMA_300M)
```

## What it does

1. Reads your CSV file
2. Generates embeddings using sentence-transformers (torch backend)
3. Writes two TSV files to `output/<model-slug>/` for Google's projector

See [PROJECT_SPEC.md](docs/PROJECT_SPEC.md) for full details.
