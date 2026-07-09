import os
import numpy as np

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    CORPUS_DIR = "/content/drive/MyDrive/iut_datathon_models/corpus"
else:
    CORPUS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "corpus",
    )


class _Retriever:
    def __init__(self):
        self.index = None
        self.texts = None
        self.model = None

    def _load(self):
        if self.index is not None:
            return
        import faiss
        from sentence_transformers import SentenceTransformer

        index_path = os.path.join(CORPUS_DIR, "bn_wiki.index")
        corpus_path = os.path.join(CORPUS_DIR, "bn_wiki.txt")

        self.index = faiss.read_index(index_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.texts = f.read().split("\n\n=====\n\n")
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def retrieve(self, query, k=1):
        self._load()
        emb = self.model.encode([query], show_progress_bar=True)
        scores, indices = self.index.search(emb.astype(np.float32), k)
        return [(self.texts[idx], float(scores[0][i])) for i, idx in enumerate(indices[0])]

    def retrieve_best(self, query):
        results = self.retrieve(query, k=1)
        return results[0][0] if results else ""


_retriever = _Retriever()


def retrieve_best_passage(prompt, response):
    query = f"{prompt} {response}"
    return _retriever.retrieve_best(query)


def retrieve_best_passage_with_score(prompt, response):
    query = f"{prompt} {response}"
    results = _retriever.retrieve(query, k=1)
    if results:
        return results[0]
    return ("", 0.0)


def retrieve_top_k(prompt, response, k=5):
    query = f"{prompt} {response}"
    results = _retriever.retrieve(query, k=k)
    passages = [p for p, _ in results]
    scores = [s for _, s in results]
    return passages, scores
