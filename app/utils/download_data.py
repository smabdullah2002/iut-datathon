import os
import sys
import argparse

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    DATA_DIR = "/content/dataset"
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
    CORPUS_DIR = "/content/corpus"
else:
    DATA_DIR = "app/dataset"
    MODELS_DIR = "models"
    CORPUS_DIR = "corpus"


def download_datasets():
    """Cache xnli_bn and other HuggingFace datasets."""
    print("Downloading xnli_bn dataset...")
    from datasets import load_dataset

    load_dataset("csebuetnlp/xnli_bn", trust_remote_code=True)
    print("xnli_bn cached successfully.")


    try:
        print("Downloading BNLI dataset...")
        load_dataset("bnli")
        print("BNLI cached.")
    except Exception as e:
        print(f"BNLI not available: {e}. Skipping.")


def download_wikipedia():
    """Download Bengali Wikipedia parquet files from HuggingFace Hub."""
    import pandas as pd
    import requests

    os.makedirs(CORPUS_DIR, exist_ok=True)
    wiki_path = os.path.join(CORPUS_DIR, "bn_wiki.txt")

    if os.path.exists(wiki_path):
        print(f"Wikipedia corpus already exists at {wiki_path}")
        return

    base = "https://huggingface.co/datasets/wikipedia/resolve/main/20220301.bn"
    files = [f"{base}/train-{i:05d}-of-00004.parquet" for i in range(4)]

    print("Downloading Bengali Wikipedia parquet files...")
    dfs = []
    for i, url in enumerate(files):
        print(f"  Downloading part {i + 1}/4...")
        dfs.append(pd.read_parquet(url))

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} articles. Writing to {wiki_path}...")

    with open(wiki_path, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            f.write(f"Title: {row['title']}\n{row['text']}\n\n=====\n\n")
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}/{len(df)} articles...")

    print(f"Saved {len(df)} articles to {wiki_path}")


def build_faiss_index():
    """Build a FAISS dense index over the Wikipedia corpus."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    os.makedirs(CORPUS_DIR, exist_ok=True)
    index_path = os.path.join(CORPUS_DIR, "bn_wiki.index")
    corpus_path = os.path.join(CORPUS_DIR, "bn_wiki.txt")

    if os.path.exists(index_path):
        print(f"FAISS index already exists at {index_path}")
        return

    if not os.path.exists(corpus_path):
        print("Corpus not found. Run download_wikipedia() first.")
        return

    print("Loading sentence-transformer model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("Reading corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = f.read().split("\n\n=====\n\n")

    print(f"Encoding {len(texts)} passages...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    print(f"FAISS index saved to {index_path} ({index.ntotal} vectors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia download")
    parser.add_argument("--skip-index", action="store_true", help="Skip FAISS index building")
    args = parser.parse_args()

    download_datasets()

    if not args.skip_wiki:
        download_wikipedia()

    if not args.skip_index:
        build_faiss_index()

    print("All data downloaded and ready.")


if __name__ == "__main__":
    main()
