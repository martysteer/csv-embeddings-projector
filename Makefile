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

# Embedding model (can override: make embed MODEL=...)
MODEL          ?= sentence-transformers/all-MiniLM-L6-v2

# Derive a filesystem-safe slug from the model name (last path component, lowercased)
MODEL_SLUG     := $(shell echo "$(MODEL)" | tr '/' '\n' | tail -1 | tr '[:upper:]' '[:lower:]' | tr '_' '-')

# Input parameters (required for embed target)
INPUT          ?=
TEXT_COL       ?=
OUTPUT         ?= $(OUTPUT_DIR)/$(MODEL_SLUG)/projector

# HuggingFace cache env vars (HF_HUB_CACHE is the current standard)
HF_ENV         := HF_HUB_CACHE=$(abspath $(MODELS_DIR)) TRANSFORMERS_VERBOSITY=error

.PHONY: all venv embed clean clean-all info download-model

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
# Model Download (optional pre-download)
# =============================================================================
download-model: $(VENV_DONE) | $(MODELS_DIR)
	@echo "Downloading embedding model: $(MODEL)"
	@$(HF_ENV) $(PYTHON) -c \
		"from sentence_transformers import SentenceTransformer; \
		SentenceTransformer('$(MODEL)')"
	@echo "✓ Model cached to $(MODELS_DIR)"

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
