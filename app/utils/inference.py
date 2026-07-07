import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.utils.preprocessing import build_input_text

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")


def load_model(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def predict_df(model, tokenizer, df, batch_size=16, max_length=256, use_retrieval=False, return_proba=False):
    device = next(model.parameters()).device

    if use_retrieval:
        from app.utils.retrieval import retrieve_best_passage
        import copy
        df = copy.deepcopy(df)
        missing = df["context"] == "[NULL]"
        retrieved = [retrieve_best_passage(p, r) for p, r in zip(df.loc[missing, "prompt_bn"], df.loc[missing, "response_bn"])]
        df.loc[missing, "context"] = retrieved
        retrieved_count = (df.loc[missing, "context"] != "").sum()
        if retrieved_count < missing.sum():
            print(f"  Retrieved {retrieved_count}/{missing.sum()} context-absent rows")

    all_preds = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        texts = [build_input_text(row, tokenizer.sep_token) for _, row in batch.iterrows()]

        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
        if return_proba:
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(probs.tolist())
        else:
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    if return_proba:
        return np.array(all_preds)  # shape (N, 2)
    return np.array(all_preds)  # shape (N,)


def write_submission(predictions, output_path="submission.csv", ids=None):
    if ids is None:
        ids = range(1, len(predictions) + 1)
    df = pd.DataFrame({"id": ids, "label": predictions})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} predictions")
