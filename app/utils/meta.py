import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from app.utils.retrieval import retrieve_best_passage_with_score, retrieve_top_k


def _static_features(df):
    features = pd.DataFrame(index=df.index)
    features["has_context"] = (df["context"] != "[NULL]").astype(int)
    features["response_len"] = df["response_bn"].str.len()

    scores = []
    for _, row in df.iterrows():
        if row["context"] == "[NULL]":
            try:
                _, top_scores = retrieve_top_k(row["prompt_bn"], row["response_bn"], k=5)
                scores.append(top_scores[0])
            except Exception:
                scores.append(0.0)
        else:
            scores.append(1.0)

    features["retrieval_score"] = scores

    if "label" in df.columns:
        features["label"] = df["label"].values
    if "band" in df.columns:
        features["band"] = df["band"].values
    return features


def extract_features(model, tokenizer, df, batch_size=16, max_length=256, use_retrieval=False):
    from app.utils.inference import predict_df
    features = _static_features(df)
    probs = predict_df(model, tokenizer, df, batch_size=batch_size, max_length=max_length,
                       use_retrieval=use_retrieval, return_proba=True)
    features["proba_1"] = probs[:, 1]
    return features


def oof_meta_features(df, tokenizer, n_splits=5, model_name="csebuetnlp/banglabert_large",
                       batch_size=8, epochs=10, lr=2e-5, max_length=256, seed=42,
                       use_retrieval=False, xnli_model=None):
    from sklearn.model_selection import StratifiedKFold
    from app.utils.preprocessing import make_samples_dataset
    from app.utils.training import get_model, train_samples
    from app.utils.inference import predict_df

    stratify_cols = df["label"].astype(str) + "_" + (df["context"] != "[NULL]").astype(str)
    if "band" in df.columns:
        stratify_cols = stratify_cols + "_" + df["band"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    static_feats = _static_features(df)
    oof_proba_1 = np.empty(len(df), dtype=np.float32)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_cols)):
        print(f"\n--- Meta CV Fold {fold + 1}/{n_splits} ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if xnli_model is not None:
            fold_model = get_model(num_labels=2, model_name=model_name)
            fold_model.load_state_dict(xnli_model.state_dict())
        else:
            fold_model = get_model(num_labels=2, model_name=model_name)

        train_ds = make_samples_dataset(tokenizer, train_df, max_length=max_length)
        val_ds = make_samples_dataset(tokenizer, val_df, max_length=max_length)

        out_dir = os.path.join("cv_cache", f"meta_fold_{seed}_{fold}")
        fold_model, _ = train_samples(
            fold_model, tokenizer, train_ds, val_ds,
            output_dir=out_dir, batch_size=batch_size, epochs=epochs, lr=lr,
            save_checkpoints=False,
        )

        val_probs = predict_df(fold_model, tokenizer, val_df, batch_size=batch_size,
                                max_length=max_length, use_retrieval=use_retrieval,
                                return_proba=True)
        oof_proba_1[val_idx] = val_probs[:, 1]
        fold_model.cpu()
        torch.cuda.empty_cache()

    train_meta = static_feats.copy()
    train_meta["proba_1"] = oof_proba_1

    return train_meta


def _best_threshold(model, X_val, y_val, bands_val=None):
    probs = model.predict_proba(X_val)[:, 1]

    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.1, 0.9, 0.025):
        preds = (probs >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average=None, zero_division=0)
        if f1[0] > best_f1:
            best_f1 = f1[0]
            best_th = th
    print(f"  Global threshold: {best_th:.3f} (f1_hallucinated: {best_f1:.4f})")

    band_thresholds = {}
    if bands_val is not None:
        print("  Per-band thresholds:")
        for band in sorted(bands_val.unique()):
            mask = (bands_val == band).values
            if mask.sum() < 5:
                band_thresholds[band] = best_th
                print(f"    {band}: {best_th:.3f} (too few samples, using global)")
                continue
            best_bth = 0.5
            best_bf1 = -1.0
            for th in np.arange(0.1, 0.9, 0.025):
                preds = (probs[mask] >= th).astype(int)
                p, r, f1, _ = precision_recall_fscore_support(y_val[mask], preds, average=None, zero_division=0)
                if f1[0] > best_bf1:
                    best_bf1 = f1[0]
                    best_bth = th
            band_thresholds[band] = best_bth
            print(f"    {band}: {best_bth:.3f} (f1_hallucinated: {best_bf1:.4f})")

    return best_th, band_thresholds


def _report_per_band(y_val, preds, bands_val, prefix=""):
    if bands_val is None:
        return
    print(f"  {prefix}Per-band F1 (hallucinated):")
    for band in sorted(bands_val.unique()):
        mask = (bands_val == band).values
        if mask.sum() < 3:
            continue
        p, r, f1, _ = precision_recall_fscore_support(y_val[mask], preds[mask], average=None, zero_division=0)
        print(f"    {band}: precision={p[0]:.3f}, recall={r[0]:.3f}, f1={f1[0]:.3f}  (n={mask.sum()})")


def train_lightgbm(train_features, val_features=None):
    import lightgbm as lgb

    has_band = "band" in train_features.columns
    X_train = train_features.drop(columns=["label"])
    y_train = train_features["label"]
    if has_band:
        X_train.pop("band")

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        num_leaves=15,
        min_child_samples=10,
        learning_rate=0.05,
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    p, r, f1, _ = precision_recall_fscore_support(y_train, train_preds, average=None, zero_division=0)
    print(f"  Train f1_hallucinated: {f1[0]:.4f}")

    threshold = 0.5
    band_thresholds = None
    if val_features is not None:
        X_val = val_features.drop(columns=["label"])
        y_val = val_features["label"]
        bands_val = X_val.pop("band") if "band" in X_val.columns else None

        val_preds = model.predict(X_val)
        p, r, f1, _ = precision_recall_fscore_support(y_val, val_preds, average=None, zero_division=0)
        print(f"  Val f1_hallucinated (th=0.5): {f1[0]:.4f}")
        _report_per_band(y_val, val_preds, bands_val, prefix="Val")

        threshold, band_thresholds = _best_threshold(model, X_val, y_val, bands_val=bands_val)

    print(f"  Feature importances: {dict(zip(X_train.columns, model.feature_importances_))}")
    return model, threshold, band_thresholds


def predict_lightgbm(model, test_features, threshold=0.5):
    X_test = test_features.copy()
    if "band" in X_test.columns:
        X_test = X_test.drop(columns=["band"])
    probs = model.predict_proba(X_test)[:, 1]
    return (probs >= threshold).astype(int)
