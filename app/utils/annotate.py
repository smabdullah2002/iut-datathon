import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

C0_KEYWORDS = [
    "সংখ্যা", "অঙ্ক", "যোগ", "বিয়োগ", "গুণ", "ভাগ", "বর্গ", "গণিত", "সমীকরণ",
    "সম্ভাবনা", "মৌলিক", "গুণিতক", "ভগ্নাংশ", "দশমিক", "বীজগণিত", "জ্যামিতি",
    "ক্ষেত্রফল", "আয়তন", "পরিমাপ", "ত্রিভুজ", "বৃত্ত", "কোণ", "লব্ধি",
    "পদার্থ", "রসায়ন", "জীববিদ্যা", "জীববিজ্ঞান", "তড়িৎ", "চুম্বক", "আলো",
    "তাপ", "বল", "গতি", "শক্তি", "চাপ", "তরঙ্গ", "কম্পন",
    "কম্পিউটার", "সিপিইউ", "CPU", "সফটওয়্যার", "অপারেটিং", "GNU", "প্রোগ্রামিং",
    "পৃথিবী", "মহাদেশ", "মহাসাগর", "দেশ", "নদী", "পর্বত", "সাগর", "দ্বীপ",
    "রাজধানী", "ভৌগলিক", "আয়তন", "জলবায়ু", "ভূগোল", "অবস্থান",
    "মানব", "দেহ", "কোষ", "জীবাণু", "রোগ", "পুষ্টি", "উদ্ভিদ", "প্রাণী",
    "সূর্য", "চাঁদ", "নক্ষত্র", "গ্রহ", "মহাকাশ", "সৌর", "জোয়ার", "ভাটা",
    "ফুটবল", "ক্রিকেট", "অলিম্পিক", "টেনিস", "ভলিবল", "বাস্কেটবল",
    "ইংরেজি", "গ্রামার", "স্পেলিং", "ভোকাবুলারি",
]

C1_KEYWORDS = [
    "বাংলা", "বাংলাদেশ", "ঢাকা", "মুক্তিযুদ্ধ", "১৯৭১", "ভাষা আন্দোলন",
    "বাঙালি", "জাতির পিতা", "শেখ মুজিব", "বঙ্গবন্ধু",
    "বাক্য", "শব্দ", "অর্থ", "সমাস", "কারক", "বিভক্তি", "ধ্বনি", "ছন্দ",
    "অলংকার", "পদ", "বানান", "শুদ্ধ", "লিপি", "সন্ধি", "উপসর্গ", "প্রত্যয়",
    "সাহিত্য", "কবি", "লেখক", "উপন্যাস", "কবিতা", "গল্প", "নাটক",
    "চলচ্চিত্র", "অভিনেতা", "পরিচালক", "চরিত্র",
    "মুঘল", "সম্রাট", "বাংলার", "নবাব", "সুলতান", "রাজা",
    "অভ্র", "কীবোর্ড", "বাংলা লিপি", "পশ্চিমবঙ্গ", "কলকাতা",
    "নজরুল", "রবীন্দ্রনাথ", "শরৎচন্দ্র", "বঙ্কিম", "মাইকেল", "মধুসূদন",
    "সংবিধান", "আইন", "ধারা", "অনুচ্ছেদ", "সরকার", "সংসদ",
    "একুশে", "পদক", "২১শে", "ফেব্রুয়ারি",
    "জান্নাতাবাদ", "সোনারগাঁও", "বাংলার ইতিহাস",
]

C2_KEYWORDS = [
    "২০২৪", "২০২৩", "২০২২", "২০২১", "২০২৫", "২০২৬", "২০২৭", "২০২৮",
    "অন্তর্বর্তীকালীন", "সাম্প্রতিক", "বর্তমান", "চলতি",
    "আওয়ামী লীগ", "বিএনপি", "জাতীয় পার্টি",
    "ড. ইউনূস", "অধ্যাপক ইউনূস",
]


def auto_fill_bands(path="cband_annotation.csv"):
    full_path = os.path.join(DATA_DIR, path)
    if not os.path.exists(full_path):
        print(f"  {full_path} not found. Run annotate first.")
        return

    df = pd.read_csv(full_path)
    empty = df["band"].isna() | (df["band"].str.strip() == "")
    if not empty.any():
        print("  All bands already filled.")
        return

    def score_band(text):
        text = str(text)
        c0 = sum(1 for kw in C0_KEYWORDS if kw in text)
        c1 = sum(1 for kw in C1_KEYWORDS if kw in text)
        c2 = sum(1 for kw in C2_KEYWORDS if kw in text)
        scores = {"C0": c0, "C1": c1, "C2": c2}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "C0"
        return best

    for idx in df[empty].index:
        text = f"{df.loc[idx, 'prompt_bn']} {df.loc[idx, 'context']}"
        df.loc[idx, "band"] = score_band(text)

    cluster_majority = df.groupby("cluster")["band"].agg(lambda x: x.value_counts().index[0])
    for idx in df[empty].index:
        cl = df.loc[idx, "cluster"]
        df.loc[idx, "band"] = cluster_majority[cl]

    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"  Auto-filled {empty.sum()} rows across {df['cluster'].nunique()} clusters.")
    print(f"  Band distribution: {df['band'].value_counts().to_dict()}")
    print("  Review and fix any misfires before using in meta.")


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

    if args.auto:
        auto_fill_bands(path=args.output or "cband_annotation.csv")
        return

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
    if getattr(args, "auto", False):
        auto_fill_bands(path=output_path)
