# =============================================================================
# CSV to Google Embeddings Projector
# =============================================================================
# Generates embeddings from CSV files for Google's Embedding Projector
# Usage: make embed INPUT=data/myfile.csv TEXT_COL=description
# =============================================================================

# --- Configuration ---
PYTHON_VERSION := 3.11
VENV_DIR       := .venv
PYTHON         := $(VENV_DIR)/bin/python
PIP            := $(VENV_DIR)/bin/pip
VENV_DONE      := $(VENV_DIR)/.done

OUTPUT_DIR     := output
MODELS_DIR     := models
SCRIPTS_DIR    := scripts

# =============================================================================
# Available Models
# =============================================================================
# Use as: make embed MODEL=$(MODEL_MINILM) ...
#      or: make embed MODEL=sentence-transformers/all-MiniLM-L6-v2 ...
#
MODEL_MINILM      := sentence-transformers/all-MiniLM-L6-v2
MODEL_GEMMA_300M  := google/embeddinggemma-300m

# Active model (default: MiniLM; override on command line)
MODEL          ?= $(MODEL_MINILM)

# Derive a filesystem-safe slug from the model name (last path component, lowercased)
MODEL_SLUG     := $(shell echo "$(MODEL)" | tr '/' '\n' | tail -1 | tr '[:upper:]' '[:lower:]' | tr '_' '-')

# Input parameters (required for embed target)
INPUT          ?=
TEXT_COL       ?=
OUTPUT         ?= $(OUTPUT_DIR)/$(MODEL_SLUG)/projector

# HuggingFace cache env vars (HF_HUB_CACHE is the current standard)
HF_ENV         := env "HF_HUB_CACHE=$(CURDIR)/$(MODELS_DIR)" TRANSFORMERS_VERBOSITY=error

.PHONY: all venv embed clean clean-all info download-model download-all-models list-models

all: venv

# =============================================================================
# Virtual Environment
# =============================================================================
venv: $(VENV_DONE)

$(VENV_DONE):
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Setting up Python virtual environment"
	@echo "═══════════════════════════════════════════════════════════════"
	@if command -v pyenv >/dev/null 2>&1; then \
		echo "Using pyenv..."; \
		pyenv install -s $(PYTHON_VERSION); \
		pyenv local $(PYTHON_VERSION); \
	fi
	@python3 -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip -q
	@$(PIP) install sentence-transformers numpy pandas -q
	@mkdir -p $(OUTPUT_DIR) $(MODELS_DIR) $(SCRIPTS_DIR)
	@touch $@
	@echo "✓ Virtual environment ready"

# =============================================================================
# Directory creation
# =============================================================================
$(MODELS_DIR) $(OUTPUT_DIR):
	@mkdir -p $@

# =============================================================================
# Model Download (optional pre-download)
# =============================================================================
download-model: $(VENV_DONE) | $(MODELS_DIR)
	@echo "Downloading embedding model: $(MODEL)"
	@$(HF_ENV) $(PYTHON) -c \
		"from sentence_transformers import SentenceTransformer; \
		SentenceTransformer('$(MODEL)', backend='torch', trust_remote_code=True, model_kwargs={'torch_dtype': 'float32'})"
	@echo "✓ Model cached to $(MODELS_DIR)"

download-all-models: $(VENV_DONE) | $(MODELS_DIR)
	@echo "Downloading all known models..."
	@$(MAKE) download-model MODEL=$(MODEL_MINILM)
	@$(MAKE) download-model MODEL=$(MODEL_GEMMA_300M)
	@echo "✓ All models cached"

list-models:
	@echo "Available models:"
	@echo "  minilm     $(MODEL_MINILM)   (no auth required)"
	@echo "  gemma-300m $(MODEL_GEMMA_300M)  (HF_TOKEN required - gated model)"
	@echo ""
	@echo "Usage:"
	@echo "  make download-model MODEL=\$$(MODEL_MINILM)"
	@echo "  make download-model MODEL=\$$(MODEL_GEMMA_300M)  # requires: export HF_TOKEN=hf_..."
	@echo "  make embed INPUT=data/file.csv TEXT_COL=title MODEL=\$$(MODEL_GEMMA_300M)"

# =============================================================================
# Embedding Generation
# =============================================================================
embed: $(VENV_DONE)
	@if [ -z "$(INPUT)" ]; then \
		echo "❌ Error: INPUT not specified"; \
		echo "Usage: make embed INPUT=data/myfile.csv TEXT_COL=description"; \
		exit 1; \
	fi
	@if [ -z "$(TEXT_COL)" ]; then \
		echo "❌ Error: TEXT_COL not specified"; \
		echo "Usage: make embed INPUT=data/myfile.csv TEXT_COL=description"; \
		exit 1; \
	fi
	@if [ ! -f "$(INPUT)" ]; then \
		echo "❌ Error: Input file '$(INPUT)' not found"; \
		exit 1; \
	fi
	@mkdir -p $(OUTPUT_DIR)/$(MODEL_SLUG)
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating embeddings"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Input:    $(INPUT)"
	@echo "  Column:   $(TEXT_COL)"
	@echo "  Model:    $(MODEL)"
	@echo "  Output:   $(OUTPUT)_vectors.tsv / $(OUTPUT)_metadata.tsv"
	@echo ""
	@$(HF_ENV) $(PYTHON) $(SCRIPTS_DIR)/embed_csv.py \
		"$(INPUT)" \
		--text-columns "$(TEXT_COL)" \
		--output "$(OUTPUT)" \
		--model "$(MODEL)"
