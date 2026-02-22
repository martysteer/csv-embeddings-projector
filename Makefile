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

# Input parameters (required for embed target)
INPUT          ?=
TEXT_COL       ?=
OUTPUT         ?= $(OUTPUT_DIR)/projector

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
	@TRANSFORMERS_CACHE=$(MODELS_DIR) $(PYTHON) -c \
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
	@mkdir -p $(OUTPUT_DIR)
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "Generating embeddings"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Input:    $(INPUT)"
	@echo "  Column:   $(TEXT_COL)"
	@echo "  Model:    $(MODEL)"
	@echo "  Output:   $(OUTPUT)_vectors.tsv / $(OUTPUT)_metadata.tsv"
	@echo ""
	@TRANSFORMERS_CACHE=$(MODELS_DIR) $(PYTHON) $(SCRIPTS_DIR)/embed_csv.py \
		"$(INPUT)" \
		--text-columns "$(TEXT_COL)" \
		--output "$(OUTPUT)" \
		--model "$(MODEL)"

# =============================================================================
# Housekeeping
# =============================================================================
clean:
	@echo "Removing output files..."
	@rm -rf $(OUTPUT_DIR)/*
	@echo "✓ Output cleaned"

clean-all:
	@echo "Removing all generated files..."
	@rm -rf $(VENV_DIR) $(OUTPUT_DIR) $(MODELS_DIR) .python-version
	@echo "✓ All files removed"

info:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "CSV Embeddings Projector"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Status:"
	@if [ -f $(VENV_DONE) ]; then echo "  ✓ Virtual environment ready"; else echo "  ✗ Run 'make venv' first"; fi
	@if [ -d $(MODELS_DIR) ] && [ "$$(ls -A $(MODELS_DIR) 2>/dev/null)" ]; then \
		echo "  ✓ Model cached"; \
	else \
		echo "  ○ Model will download on first use"; \
	fi
	@echo ""
	@echo "Output files:"
	@if [ -d $(OUTPUT_DIR) ] && [ "$$(ls -A $(OUTPUT_DIR) 2>/dev/null)" ]; then \
		ls -lh $(OUTPUT_DIR)/*.tsv 2>/dev/null | awk '{print "  " $$9 " (" $$5 ")"}' || echo "  (none)"; \
	else \
		echo "  (none)"; \
	fi
	@echo ""
	@echo "Usage:"
	@echo "  make embed INPUT=data/myfile.csv TEXT_COL=description"
	@echo ""
	@echo "Then upload to: https://projector.tensorflow.org/"
