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
    """Download Bengali Wikipedia via HuggingFace datasets (rejauldu/bengali-wikipedia)."""
    from datasets import load_dataset

    os.makedirs(CORPUS_DIR, exist_ok=True)
    wiki_path = os.path.join(CORPUS_DIR, "bn_wiki.txt")

    if os.path.exists(wiki_path):
        print(f"Wikipedia corpus already exists at {wiki_path}")
        return

    print("Downloading rejauldu/bengali-wikipedia via HuggingFace datasets...")
    ds = load_dataset("rejauldu/bengali-wikipedia", split="train")
    print(f"Loaded {len(ds)} articles. Writing to {wiki_path}...")

    with open(wiki_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            f.write(f"{example['text']}\n\n=====\n\n")
            if (i + 1) % 50000 == 0:
                print(f"  Written {i + 1}/{len(ds)} articles...")

    print(f"Saved {len(ds)} articles to {wiki_path}")


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
