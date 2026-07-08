import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from app.utils.retrieval import retrieve_best_passage_with_score


def _static_features(df):
    features = pd.DataFrame(index=df.index)
    features["has_context"] = (df["context"] != "[NULL]").astype(int)
    features["response_len"] = df["response_bn"].str.len()
    scores = []
    for _, row in df.iterrows():
        if row["context"] == "[NULL]":
            _, score = retrieve_best_passage_with_score(row["prompt_bn"], row["response_bn"])
            scores.append(score)
        else:
            scores.append(1.0)
    features["retrieval_score"] = scores
    if "label" in df.columns:
        features["label"] = df["label"].values
    return features


def extract_features(model, tokenizer, df, batch_size=16, max_length=256, use_retrieval=False):
    from app.utils.inference import predict_df
    features = _static_features(df)
    probs = predict_df(model, tokenizer, df, batch_size=batch_size, max_length=max_length,
                       use_retrieval=use_retrieval, return_proba=True)
    features["proba_0"] = probs[:, 0]
    features["proba_1"] = probs[:, 1]
    return features


def oof_meta_features(df, tokenizer, n_splits=5, model_name="csebuetnlp/banglabert_large",
                       batch_size=8, epochs=10, lr=2e-5, max_length=256, seed=42,
                       use_retrieval=False, xnli_model=None):
    from sklearn.model_selection import StratifiedKFold
    from app.utils.preprocessing import make_samples_dataset
    from app.utils.training import get_model, train_samples
    from app.utils.inference import predict_df

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stratify_cols = df["label"].astype(str) + "_" + (df["context"] != "[NULL]").astype(str)

    static_feats = _static_features(df)
    oof_proba_0 = np.empty(len(df), dtype=np.float32)
    oof_proba_1 = np.empty(len(df), dtype=np.float32)
    fold_models = []

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

        out_dir = os.path.join("cv_cache", f"meta_fold_{fold}")
        fold_model, _ = train_samples(
            fold_model, tokenizer, train_ds, val_ds,
            output_dir=out_dir, batch_size=batch_size, epochs=epochs, lr=lr,
        )

        val_probs = predict_df(fold_model, tokenizer, val_df, batch_size=batch_size,
                                max_length=max_length, use_retrieval=use_retrieval,
                                return_proba=True)
        oof_proba_0[val_idx] = val_probs[:, 0]
        oof_proba_1[val_idx] = val_probs[:, 1]
        fold_models.append(fold_model)

    train_meta = static_feats.copy()
    train_meta["proba_0"] = oof_proba_0
    train_meta["proba_1"] = oof_proba_1

    return train_meta, fold_models


def train_lightgbm(train_features, val_features=None):
    import lightgbm as lgb

    X_train = train_features.drop(columns=["label"])
    y_train = train_features["label"]

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

    if val_features is not None:
        X_val = val_features.drop(columns=["label"])
        y_val = val_features["label"]
        val_preds = model.predict(X_val)
        p, r, f1, _ = precision_recall_fscore_support(y_val, val_preds, average=None, zero_division=0)
        print(f"  Val f1_hallucinated:   {f1[0]:.4f}")

    print(f"  Feature importances: {dict(zip(X_train.columns, model.feature_importances_))}")
    return model


def predict_lightgbm(model, test_features):
    return model.predict(test_features)
