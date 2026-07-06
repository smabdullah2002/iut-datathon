import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_df(model, tokenizer, df, batch_size=16, max_length=256):
    device = next(model.parameters()).device
    all_preds = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        texts = []
        for _, row in batch.iterrows():
            if row["context"] == "[NULL]":
                text = f"{tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
            else:
                text = f"{row['context']} {tokenizer.sep_token} {row['prompt_bn']} {tokenizer.sep_token} {row['response_bn']}"
            texts.append(text)

        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())

    return np.array(all_preds)


def write_submission(predictions, output_path="submission.csv"):
    df = pd.DataFrame({"id": range(1, len(predictions) + 1), "label": predictions})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} predictions")
