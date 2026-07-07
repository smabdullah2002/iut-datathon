import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def extract_features(model, tokenizer, df, batch_size=16, max_length=256, use_retrieval=False):
    from app.utils.inference import predict_df
    from app.utils.retrieval import retrieve_best_passage_with_score

    features = pd.DataFrame(index=df.index)
    has_orig_context = (df["context"] != "[NULL"])
    features["has_context"] = has_orig_context.astype(int)
    features["response_len"] = df["response_bn"].str.len()

    probs = predict_df(model, tokenizer, df, batch_size=batch_size, max_length=max_length, return_proba=True)
    features["proba_0"] = probs[:, 0]
    features["proba_1"] = probs[:, 1]

    scores = []
    for _, row in df.iterrows():
        if row["context"] == "[NULL]":
            _, score = retrieve_best_passage_with_score(row["prompt_bn"], row["response_bn"])
            scores.append(score)
        else:
            scores.append(1.0)
    features["retrieval_score"] = scores
    features["has_retrieval"] = (~has_orig_context).astype(int)

    if "label" in df.columns:
        features["label"] = df["label"].values

    return features


def train_lightgbm(train_features, val_features=None):
    import lightgbm as lgb

    X_train = train_features.drop(columns=["label"])
    y_train = train_features["label"]

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
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
