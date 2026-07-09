import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")


def set_seed(seed=42):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cband(path="cband_annotation.csv"):
    full_path = os.path.join(DATA_DIR, path)
    if not os.path.exists(full_path):
        return None
    df = pd.read_csv(full_path)
    df["band"] = df["band"].str.strip().str.upper()
    invalid = ~df["band"].isin(["C0", "C1", "C2"])
    if invalid.any():
        print(f"Warning: {invalid.sum()} rows in {path} have invalid or empty bands. Returning None.")
        return None
    return df[["id", "band"]]


def load_samples(load_bands=True):
    path = os.path.join(DATA_DIR, "samples.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["response_bn"] = df["response_bn"].astype(str)
    df["has_context"] = (df["context"] != "[NULL]").astype(int)
    if load_bands:
        cband = load_cband()
        if cband is not None:
            df = df.merge(cband, left_index=True, right_on="id", how="left")
            df["band"] = df["band"].fillna("C0")
    return df


def load_test_set():
    path = os.path.join(DATA_DIR, "test set.csv")
    df = pd.read_csv(path)
    df["response_bn"] = df["response_bn"].astype(str)
    df["has_context"] = (df["context"] != "[NULL]").astype(int)
    return df


def split_samples(df, val_size=0.2, random_state=42):
    stratify_cols = df["label"].astype(str) + "_" + df["has_context"].astype(str)
    if "band" in df.columns:
        stratify_cols = stratify_cols + "_" + df["band"].astype(str)
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=stratify_cols,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_xnli_bn(split="train"):
    ds = load_dataset("csebuetnlp/xnli_bn", split=split, trust_remote_code=True)
    df = pd.DataFrame(ds)
    df.rename(columns={"sentence1": "premise", "sentence2": "hypothesis"}, inplace=True)
    df = df[df["label"] != 2].reset_index(drop=True)
    df["label"] = (df["label"] == 1).astype(int)
    return df


def get_tokenizer(model_name="csebuetnlp/banglabert_large"):
    return AutoTokenizer.from_pretrained(model_name)


def _patch_torchvision():
    import torchvision
    if not hasattr(torchvision.io, "VideoReader"):
        class _DummyVideoReader:
            pass
        torchvision.io.VideoReader = _DummyVideoReader

_patch_torchvision()


def build_input_text(row, sep_token):
    if row["context"] == "[NULL]":
        return f"{sep_token} {row['prompt_bn']} {sep_token} {row['response_bn']}"
    return f"{row['context']} {sep_token} {row['prompt_bn']} {sep_token} {row['response_bn']}"


def make_samples_dataset(tokenizer, df, max_length=256):
    texts = [build_input_text(row, tokenizer.sep_token) for _, row in df.iterrows()]
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": df["label"].tolist(),
    })
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def make_inference_dataset(tokenizer, df, max_length=256):
    texts = [build_input_text(row, tokenizer.sep_token) for _, row in df.iterrows()]
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    })
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return ds


def make_xnli_dataset(tokenizer, df, max_length=256, batch_size=1000):
    ds = Dataset.from_pandas(df[["premise", "hypothesis", "label"]])

    def tokenize_batch(batch):
        texts = [f"{p} {tokenizer.sep_token} {h}" for p, h in zip(batch["premise"], batch["hypothesis"])]
        result = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        return {"input_ids": result["input_ids"], "attention_mask": result["attention_mask"], "labels": batch["label"]}

    ds = ds.map(tokenize_batch, batched=True, batch_size=batch_size, remove_columns=["premise", "hypothesis"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds
