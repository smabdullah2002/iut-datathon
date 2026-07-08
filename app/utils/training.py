import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
else:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "models")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    return {
        "f1_hallucinated": f1[0],
        "precision_hallucinated": precision[0],
        "recall_hallucinated": recall[0],
        "f1_faithful": f1[1],
    }


def get_model(num_labels=2, model_name="csebuetnlp/banglabert_large"):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


class _BestModelCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.best_f1 = -1.0
        self.best_state_dict = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        f1 = metrics.get("eval_f1_hallucinated", -1.0)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def train_xnli_warmup(model, tokenizer, xnli_dataset, output_dir=None, batch_size=16, epochs=2, lr=3e-5):
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "xnli_warmup")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="no",
        logging_steps=500,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=xnli_dataset,
    )

    trainer.train()
    return model


def train_samples(model, tokenizer, train_dataset, val_dataset, output_dir=None, batch_size=8, epochs=30, lr=2e-5, save_checkpoints=True):
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "samples_finetune")

    if save_checkpoints:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_hallucinated",
            greater_is_better=True,
            logging_steps=10,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        return trainer.model, metrics
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_steps=10,
            remove_unused_columns=False,
            report_to="none",
        )

        callback = _BestModelCallback(model)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[callback],
        )

        trainer.train()
        metrics = trainer.evaluate()

        if callback.best_state_dict is not None:
            model.load_state_dict(callback.best_state_dict)
            print(f"  Restored best model (f1_hallucinated: {callback.best_f1:.4f})")

        return model, metrics


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")


def cross_validate(df, tokenizer, n_splits=5, model_name="csebuetnlp/banglabert_large", batch_size=8, epochs=30, lr=2e-5, max_length=256, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from app.utils.preprocessing import make_samples_dataset

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stratify_cols = df["label"].astype(str) + "_" + df["has_context"].astype(str)

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, stratify_cols)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        fold_model = get_model(num_labels=2, model_name=model_name)
        train_ds = make_samples_dataset(tokenizer, train_df, max_length=max_length)
        val_ds = make_samples_dataset(tokenizer, val_df, max_length=max_length)

        fold_args = TrainingArguments(
            output_dir=os.path.join("cv_cache", f"fold_{fold}"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=fold_model,
            args=fold_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        all_metrics.append(metrics)
        f1 = metrics["eval_f1_hallucinated"]
        print(f"  f1_hallucinated = {f1:.4f}")

    f1s = [m["eval_f1_hallucinated"] for m in all_metrics]
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    print(f"\n=== CV Summary ({n_splits}-fold) ===")
    print(f"  Per-fold: {[f'{f:.4f}' for f in f1s]}")
    print(f"  Mean f1_hallucinated: {mean_f1:.4f} ± {std_f1:.4f}")

    return {"per_fold": f1s, "mean": mean_f1, "std": std_f1}
