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


def load_samples():
    path = os.path.join(DATA_DIR, "samples.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["response_bn"] = df["response_bn"].astype(str)
    df["has_context"] = (df["context"] != "[NULL]").astype(int)
    return df


def load_test_set():
    path = os.path.join(DATA_DIR, "test set.csv")
    df = pd.read_csv(path)
    df["response_bn"] = df["response_bn"].astype(str)
    df["has_context"] = (df["context"] != "[NULL]").astype(int)
    return df


def split_samples(df, val_size=0.2, random_state=42):
    stratify_cols = df["label"].astype(str) + "_" + df["has_context"].astype(str)
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


def get_tokenizer():
    return AutoTokenizer.from_pretrained("csebuetnlp/banglabert_large")


def make_samples_dataset(tokenizer, df, max_length=256):
    texts = []
    for _, row in df.iterrows():
        if row["context"] == "[NULL]":
            text = f"{tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
        else:
            text = f"{row['context']} {tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
        texts.append(text)
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    ds = Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": df["label"].tolist(),
    })
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def make_inference_dataset(tokenizer, df, max_length=256):
    texts = []
    for _, row in df.iterrows():
        if row["context"] == "[NULL]":
            text = f"{tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
        else:
            text = f"{row['context']} {tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
        texts.append(text)
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
