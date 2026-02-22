#!/usr/bin/env python3
"""
Embed CSV text columns and export to Google Embedding Projector format.

Generates two TSV files:
  - {output}_vectors.tsv: Tab-separated embedding vectors (no header)
  - {output}_metadata.tsv: Tab-separated metadata with headers

Usage:
  python embed_csv.py data/myfile.csv --text-columns description --output projector
  python embed_csv.py data/myfile.csv --text-columns "title,description" --output projector
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file with automatic encoding detection."""
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='latin-1')


def combine_text_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Combine multiple columns into single text strings."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in columns:
            val = row.get(col, '')
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        texts.append(' '.join(parts))
    return texts


def generate_embeddings(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for text list with progress."""
    print(f"📖 Loading model: {model_name}")
    model = SentenceTransformer(
        model_name,
        backend="torch",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "float32"},
    )
    
    print(f"🔢 Generating embeddings for {len(texts)} records...")
    embeddings = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
        
        done = min(i + batch_size, total)
        percent = int(done * 100 / total)
        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
        print(f"\r  Progress: [{bar}] {percent}% ({done}/{total})", end='', flush=True)
    
    print()  # newline after progress
    return np.vstack(embeddings)


def save_vectors(embeddings: np.ndarray, output_path: Path) -> None:
    """Save embeddings as TSV (no header, tab-separated floats)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for embedding in embeddings:
            line = '\t'.join(str(val) for val in embedding)
            f.write(line + '\n')
    print(f"✅ Vectors saved: {output_path}")


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """Save metadata as TSV with headers."""
    # Escape tabs and newlines in string columns
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('').astype(str)
            df_clean[col] = df_clean[col].str.replace('\t', '\\t', regex=False)
            df_clean[col] = df_clean[col].str.replace('\n', '\\n', regex=False)
    
    df_clean.to_csv(output_path, sep='\t', index=False)
    print(f"✅ Metadata saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Embed CSV text columns for Google Embedding Projector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embed_csv.py data/books.csv --text-columns title
  python embed_csv.py data/articles.csv --text-columns "title,abstract" --output research

Output:
  {output}_vectors.tsv  - Upload as vectors
  {output}_metadata.tsv - Upload as metadata
  
Upload both to: https://projector.tensorflow.org/
        """
    )
    
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument(
        '--text-columns', '-t',
        default=None,
        help='Column(s) to embed, comma-separated (default: all columns)'
    )
    parser.add_argument(
        '--output', '-o',
        default='projector',
        help='Output file prefix (default: projector)'
    )
    parser.add_argument(
        '--model', '-m',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size for embedding (default: 64)'
    )
    
    args = parser.parse_args()

    # Paths
    input_path = Path(args.input)
    vectors_path = Path(f"{args.output}_vectors.tsv")
    metadata_path = Path(f"{args.output}_metadata.tsv")

    # Load CSV
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    df = load_csv(str(input_path))
    print(f"📄 Loaded {len(df)} rows, {len(df.columns)} columns")

    # Resolve text columns (default: all)
    if args.text_columns:
        text_columns = [c.strip() for c in args.text_columns.split(',')]
    else:
        text_columns = list(df.columns)
        print(f"   No TEXT_COL specified — using all columns: {', '.join(text_columns)}")

    print("═" * 60)
    print("CSV to Google Embedding Projector")
    print("═" * 60)
    print(f"  Input:    {input_path}")
    print(f"  Columns:  {', '.join(text_columns)}")
    print(f"  Model:    {args.model}")
    print()
    
    # Validate columns exist
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        print(f"❌ Error: Column(s) not found: {', '.join(missing)}")
        print(f"   Available: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Combine text columns
    texts = combine_text_columns(df, text_columns)
    
    # Check for empty texts
    empty_count = sum(1 for t in texts if not t.strip())
    if empty_count > 0:
        print(f"⚠️  Warning: {empty_count} rows have empty text")
    
    # Generate embeddings
    embeddings = generate_embeddings(texts, args.model, args.batch_size)
    print(f"   Dimensions: {embeddings.shape[1]}")
    
    # Ensure output directory exists
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    print()
    save_vectors(embeddings, vectors_path)
    save_metadata(df, metadata_path)
    
    print()
    print("═" * 60)
    print("✅ Done! Upload to https://projector.tensorflow.org/")
    print("═" * 60)
    print(f"   1. Vectors:  {vectors_path}")
    print(f"   2. Metadata: {metadata_path}")


if __name__ == '__main__':
    main()
