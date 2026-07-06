import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
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
        tokenizer=tokenizer,
    )

    trainer.train()
    return model


def train_samples(model, tokenizer, train_dataset, val_dataset, output_dir=None, batch_size=8, epochs=30, lr=2e-5):
    if output_dir is None:
        output_dir = os.path.join(MODELS_DIR, "samples_finetune")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_hallucinated",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.model, trainer.state.best_model_checkpoint


def save_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")
