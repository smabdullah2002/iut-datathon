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

    load_dataset("csebuetnlp/xnli_bn", revision="refs/convert/parquet")
    print("xnli_bn cached successfully.")


    try:
        print("Downloading BNLI dataset...")
        load_dataset("bnli", trust_remote_code=True)
        print("BNLI cached.")
    except Exception:
        print("BNLI not found on HF, skipping.")


def download_wikipedia():
    """Download Bengali Wikipedia dump for retrieval corpus."""
    import urllib.request
    import bz2

    os.makedirs(CORPUS_DIR, exist_ok=True)

    wiki_path = os.path.join(CORPUS_DIR, "bn_wiki.txt")
    if os.path.exists(wiki_path):
        print(f"Wikipedia corpus already exists at {wiki_path}")
        return

    dump_url = "https://dumps.wikimedia.org/bnwiki/latest/bnwiki-latest-pages-articles.xml.bz2"
    dump_path = os.path.join(CORPUS_DIR, "bnwiki-latest-pages-articles.xml.bz2")

    if not os.path.exists(dump_path):
        print(f"Downloading Bengali Wikipedia dump from {dump_url}...")
        urllib.request.urlretrieve(dump_url, dump_path)
        print("Download complete.")
    else:
        print("Wikipedia dump already downloaded.")

    print("Extracting articles (this may take a while)...")
    with bz2.open(dump_path, "rb") as f:
        content = f.read().decode("utf-8", errors="replace")

    import xml.etree.ElementTree as ET
    import re

    root = ET.fromstring("<?xml version='1.0' encoding='utf-8'?><mediawiki>" + content.split("<mediawiki>")[1].rsplit("</mediawiki>")[0] + "</mediawiki>")
    articles = []
    for page in root.findall(".//page"):
        title = page.findtext("title", "")
        text = page.findtext(".//text", "")
        if text.strip():
            articles.append(f"Title: {title}\n{text.strip()}")

    with open(wiki_path, "w", encoding="utf-8") as f:
        f.write("\n\n=====\n\n".join(articles))

    print(f"Extracted {len(articles)} articles to {wiki_path}")


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
