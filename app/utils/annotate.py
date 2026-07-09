import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def embed_samples(df, model_name="csebuetnlp/banglabert_large"):
    from transformers import AutoModel
    from app.utils.preprocessing import get_tokenizer, build_input_text
    import torch

    tokenizer = get_tokenizer(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    texts = [build_input_text(row, tokenizer.sep_token) for _, row in df.iterrows()]

    all_embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.concatenate(all_embeddings, axis=0)


def cluster_embeddings(embeddings, n_clusters=10, seed=42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km


def write_annotation_template(df, cluster_labels, path="cband_annotation.csv"):
    output = pd.DataFrame({
        "id": range(len(df)),
        "cluster": cluster_labels,
        "prompt_bn": df["prompt_bn"].values,
        "response_bn": df["response_bn"].values,
        "context": df["context"].values,
        "label": df["label"].values,
        "band": "",
    })
    output = output.sort_values(["cluster", "id"]).reset_index(drop=True)
    full_path = os.path.join(DATA_DIR, path)
    output.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"Annotation template saved to {full_path}")
    print(f"  {len(output)} rows in {len(np.unique(cluster_labels))} clusters")
    print("  Fill the 'band' column with C0, C1, or C2 for each row.")


def load_annotations(path="cband_annotation.csv"):
    full_path = os.path.join(DATA_DIR, path)
    if not os.path.exists(full_path):
        return None
    df = pd.read_csv(full_path)
    df["band"] = df["band"].str.strip().str.upper()
    invalid = ~df["band"].isin(["C0", "C1", "C2"])
    if invalid.any():
        print(f"Warning: {invalid.sum()} rows have invalid or empty bands.")
        return None
    return df[["id", "band"]]


def main(args):
    from app.utils.preprocessing import load_samples

    print("Loading samples...")
    df = load_samples(load_bands=False)

    print("Generating BanglaBERT embeddings...")
    embeddings = embed_samples(df)
    print(f"  Embeddings shape: {embeddings.shape}")

    n_clusters = min(args.clusters, max(5, len(df) // 10))
    print(f"Clustering into {n_clusters} clusters...")
    labels, _ = cluster_embeddings(embeddings, n_clusters=n_clusters)

    output_path = args.output or "cband_annotation.csv"
    write_annotation_template(df, labels, path=output_path)
