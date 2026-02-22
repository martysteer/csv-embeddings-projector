# CSV to Google Embedding Projector
# Usage: make embed INPUT=data/myfile.csv TEXT_COL=description

PYTHON_VERSION := 3.11
VENV_DIR       := .venv
PYTHON         := $(VENV_DIR)/bin/python
PIP            := $(VENV_DIR)/bin/pip
VENV_DONE      := $(VENV_DIR)/.done

OUTPUT_DIR     := output
MODELS_DIR     := models

# Models — override on the command line with MODEL=...
MODEL_MINILM     := sentence-transformers/all-MiniLM-L6-v2
MODEL_GEMMA_300M := google/embeddinggemma-300m
MODEL            ?= $(MODEL_MINILM)

# Filesystem-safe slug from the last component of the model name
MODEL_SLUG := $(shell echo "$(MODEL)" | tr '/' '\n' | tail -1 | tr '[:upper:]' '[:lower:]' | tr '_' '-')

OUTPUT   ?= $(OUTPUT_DIR)/$(MODEL_SLUG)/projector
HF_ENV   := env "HF_HUB_CACHE=$(CURDIR)/$(MODELS_DIR)" TRANSFORMERS_VERBOSITY=error

# Clustering levels — number of clusters to cut the dendrogram at
LEVELS         ?= 3 5 10 20
VECTORS_TSV    ?= $(OUTPUT_DIR)/$(MODEL_SLUG)/projector_vectors.tsv
METADATA_TSV   ?= $(OUTPUT_DIR)/$(MODEL_SLUG)/projector_metadata.tsv

# Facet compression
FACET_COLS     ?=
TOP_N          ?= 10

.PHONY: all venv download-model download-all-models embed clusters facets clean clean-all help

all: venv

# -- Virtual environment ------------------------------------------------------

venv: $(VENV_DONE)

$(VENV_DONE):
	@if command -v pyenv >/dev/null 2>&1; then \
		pyenv install -s $(PYTHON_VERSION); \
		pyenv local $(PYTHON_VERSION); \
	fi
	@python3 -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip -q
	@$(PIP) install sentence-transformers numpy pandas scipy -q
	@mkdir -p $(OUTPUT_DIR) $(MODELS_DIR)
	@touch $@
	@echo "✓ Virtual environment ready"

# -- Models -------------------------------------------------------------------

download-model: $(VENV_DONE)
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading: $(MODEL)"
	@$(HF_ENV) $(PYTHON) -c \
		"from sentence_transformers import SentenceTransformer; \
		SentenceTransformer('$(MODEL)', backend='torch', trust_remote_code=True, model_kwargs={'torch_dtype': 'float32'})"
	@echo "✓ Cached to $(MODELS_DIR)"

download-all-models: $(VENV_DONE)
	@for m in $(MODEL_MINILM) $(MODEL_GEMMA_300M); do \
		$(MAKE) download-model MODEL=$$m; \
	done

# -- Embed --------------------------------------------------------------------

embed: $(VENV_DONE)
	@if [ -z "$(INPUT)" ]; then \
		echo "❌  Usage: make embed INPUT=data/file.csv [TEXT_COL=description] [MODEL=...]"; \
		exit 1; \
	fi
	@if [ ! -f "$(INPUT)" ]; then \
		echo "❌  Input file not found: $(INPUT)"; \
		exit 1; \
	fi
	@mkdir -p $(OUTPUT_DIR)/$(MODEL_SLUG)
	@$(HF_ENV) $(PYTHON) scripts/embed_csv.py \
		"$(INPUT)" \
		$(if $(TEXT_COL),--text-columns "$(TEXT_COL)") \
		--output "$(OUTPUT)" \
		--model "$(MODEL)"

# -- Cluster ------------------------------------------------------------------

clusters: $(VENV_DONE)
	@if [ ! -f "$(VECTORS_TSV)" ]; then \
		echo "❌  Vectors not found: $(VECTORS_TSV)"; \
		echo "   Run 'make embed INPUT=...' first."; \
		exit 1; \
	fi
	@$(PYTHON) scripts/cluster_embeddings.py \
		"$(VECTORS_TSV)" \
		--metadata "$(METADATA_TSV)" \
		--levels $(LEVELS)

# -- Facets -------------------------------------------------------------------

facets: $(VENV_DONE)
	@if [ -z "$(FACET_COLS)" ]; then \
		echo "❌  Usage: make facets FACET_COLS=publisher,genre [TOP_N=10] [MODEL=...]"; \
		exit 1; \
	fi
	@if [ ! -f "$(METADATA_TSV)" ]; then \
		echo "❌  Metadata not found: $(METADATA_TSV)"; \
		echo "   Run 'make embed INPUT=...' first."; \
		exit 1; \
	fi
	@$(PYTHON) scripts/compress_metadata.py \
		"$(METADATA_TSV)" \
		--columns "$(FACET_COLS)" \
		--top-n $(TOP_N)

# -- Housekeeping -------------------------------------------------------------

clean:
	rm -rf $(OUTPUT_DIR)

clean-all: clean
	rm -rf $(VENV_DIR) $(MODELS_DIR)

# -- Help ---------------------------------------------------------------------

help:
	@echo "Usage: make <target> [VARIABLE=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  venv                 Create Python virtual environment"
	@echo "  download-model       Download MODEL to models/ cache"
	@echo "  download-all-models  Download all known models"
	@echo "  embed                Generate embeddings from a CSV"
	@echo "  clusters             Hierarchical clustering of embedded vectors"
	@echo "  facets               Compress high-cardinality columns for colour-by"
	@echo "  clean                Remove output/"
	@echo "  clean-all            Remove output/, models/, .venv/"
	@echo "  help                 Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL        HuggingFace model ID (default: $(MODEL_MINILM))"
	@echo "  INPUT        Path to input CSV (required for embed)"
	@echo "  TEXT_COL     Column(s) to embed, comma-separated (default: all columns)"
	@echo "  OUTPUT       Output prefix (default: output/<model-slug>/projector)"
	@echo "  LEVELS       Cluster counts for dendrogram cuts (default: 3 5 10 20)"
	@echo "  VECTORS_TSV  Vectors file for clusters (default: output/<slug>/projector_vectors.tsv)"
	@echo "  METADATA_TSV Metadata file for clusters (default: output/<slug>/projector_metadata.tsv)"
	@echo "  FACET_COLS   Comma-separated columns to compress (required for facets)"
	@echo "  TOP_N        Number of top values to keep per column (default: 10)"
	@echo ""
	@echo "Available models:"
	@echo "  $(MODEL_MINILM)   no auth required"
	@echo "  $(MODEL_GEMMA_300M)            gated — export HF_TOKEN=hf_... first"
	@echo ""
	@echo "Examples:"
	@echo "  make embed INPUT=data/sample.csv TEXT_COL=description"
	@echo "  make clusters                                  # default 3 5 10 20 levels"
	@echo "  make clusters LEVELS='5 10 25 50'             # custom levels"
	@echo "  make facets FACET_COLS=publisher,genre        # top 10 + Other"
	@echo "  make facets FACET_COLS=publisher TOP_N=8      # top 8 + Other"
	@echo "  make embed INPUT=data/sample.csv TEXT_COL=description MODEL=$(MODEL_GEMMA_300M)"
	@echo "  make download-model MODEL=$(MODEL_GEMMA_300M)"
