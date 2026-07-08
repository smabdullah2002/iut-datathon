import os
import argparse

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    BASE_DIR = "/content/iut-datathon"
    DATA_DIR = "/content/iut-datathon/app/dataset"
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
    CORPUS_DIR = "/content/corpus"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    CORPUS_DIR = os.path.join(BASE_DIR, "..", "corpus")


def train(args):
    from app.utils.preprocessing import (
        load_samples, split_samples, load_xnli_bn, get_tokenizer,
        make_samples_dataset, make_xnli_dataset, set_seed,
    )
    from app.utils.training import get_model, train_xnli_warmup, train_samples, save_model

    set_seed(args.seed)

    print("Loading samples...")
    df = load_samples()

    tokenizer = get_tokenizer(model_name=args.model_name)

    if args.retrieve:
        print("Augmenting context-absent rows with Wikipedia retrieval...")
        from app.utils.retrieval import retrieve_best_passage
        missing = df["context"] == "[NULL]"
        retrieved = [retrieve_best_passage(p, r) for p, r in zip(df.loc[missing, "prompt_bn"], df.loc[missing, "response_bn"])]
        df.loc[missing, "context"] = retrieved
        filled = (df.loc[missing, "context"] != "").sum()
        print(f"  Retrieved {filled}/{missing.sum()} passages")

    if args.cv > 0:
        print(f"Running {args.cv}-fold cross-validation...")
        from app.utils.training import cross_validate
        cv_results = cross_validate(
            df, tokenizer, n_splits=args.cv, model_name=args.model_name,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            max_length=args.max_length, seed=args.seed,
        )

    train_df, val_df = split_samples(df, val_size=args.val_split, random_state=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    model = get_model(num_labels=2, model_name=args.model_name)

    if not args.skip_xnli:
        print("Loading xnli_bn for warm-up...")
        xnli_df = load_xnli_bn()
        print(f"  {len(xnli_df)} NLI pairs loaded (contradiction + entailment)")
        xnli_dataset = make_xnli_dataset(tokenizer, xnli_df, max_length=args.max_length)
        print(f"  Warm-up training for {args.xnli_epochs} epochs...")
        model = train_xnli_warmup(
            model, tokenizer, xnli_dataset,
            batch_size=args.batch_size, epochs=args.xnli_epochs, lr=args.xnli_lr,
        )
        print("  Warm-up complete.")

    train_dataset = make_samples_dataset(tokenizer, train_df, max_length=args.max_length)
    val_dataset = make_samples_dataset(tokenizer, val_df, max_length=args.max_length)

    print(f"Fine-tuning for {args.epochs} epochs...")
    model, final_metrics = train_samples(
        model, tokenizer, train_dataset, val_dataset,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
    )
    print(f"  Best val f1_hallucinated: {final_metrics.get('eval_f1_hallucinated', 'N/A'):.4f}")

    save_path = os.path.join(MODELS_DIR, "best_model")
    save_model(model, tokenizer, save_path)
    print("Training complete.")


def predict(args):
    import numpy as np
    from app.utils.preprocessing import load_test_set, get_tokenizer
    from app.utils.inference import load_model, predict_df, write_submission

    print("Loading model...")
    model_dir = args.model_dir or os.path.join(MODELS_DIR, "best_model")
    model, tokenizer = load_model(model_dir)
    print(f"Model loaded from {model_dir}")

    print("Loading test set...")
    test_df = load_test_set()
    print(f"  {len(test_df)} test samples")

    ids = test_df["id"].values

    print("Running inference...")
    predictions = predict_df(model, tokenizer, test_df, batch_size=args.batch_size, max_length=args.max_length, use_retrieval=args.retrieve)

    vals, counts = np.unique(predictions, return_counts=True)
    label_dist = dict(zip(vals, counts))
    print(f"  Predictions: {label_dist}")

    write_submission(predictions, output_path=args.output, ids=ids)


def meta(args):
    import numpy as np
    from app.utils.preprocessing import (
        load_samples, load_test_set, get_tokenizer,
        split_samples, load_xnli_bn, make_xnli_dataset, make_samples_dataset,
    )
    from app.utils.inference import write_submission
    from app.utils.meta import oof_meta_features, extract_features, train_lightgbm, predict_lightgbm
    from app.utils.training import get_model, train_xnli_warmup, train_samples
    from sklearn.model_selection import train_test_split

    print("Loading datasets...")
    train_df = load_samples()
    test_df = load_test_set()

    tokenizer = get_tokenizer(model_name=args.model_name)
    use_retrieval = args.retrieve
    if use_retrieval:
        print("Using retrieval for context-absent rows")

    base_model = get_model(num_labels=2, model_name=args.model_name)
    if not args.skip_xnli:
        print("Loading xnli_bn for warmup...")
        xnli_df = load_xnli_bn()
        xnli_dataset = make_xnli_dataset(tokenizer, xnli_df, max_length=args.max_length)
        print(f"  XNLI warmup for {args.xnli_epochs} epochs...")
        base_model = train_xnli_warmup(
            base_model, tokenizer, xnli_dataset,
            batch_size=args.batch_size, epochs=args.xnli_epochs, lr=args.xnli_lr,
        )

    n_splits = max(2, args.cv)
    print(f"\nGenerating OOF probabilities ({n_splits}-fold CV)...")
    train_meta, fold_models = oof_meta_features(
        train_df, tokenizer, n_splits=n_splits,
        model_name=args.model_name,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        max_length=args.max_length, seed=args.seed,
        use_retrieval=use_retrieval,
        xnli_model=base_model if not args.skip_xnli else None,
    )

    print("\nTraining final model on all data...")
    train_fold, val_fold = split_samples(train_df, val_size=args.val_split, random_state=args.seed)
    train_ds = make_samples_dataset(tokenizer, train_fold, max_length=args.max_length)
    val_ds = make_samples_dataset(tokenizer, val_fold, max_length=args.max_length)

    final_model = get_model(num_labels=2, model_name=args.model_name)
    if not args.skip_xnli:
        final_model.load_state_dict(base_model.state_dict())
    final_model, final_metrics = train_samples(
        final_model, tokenizer, train_ds, val_ds,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
    )

    print("\nExtracting test features...")
    test_feats = extract_features(
        final_model, tokenizer, test_df,
        use_retrieval=use_retrieval,
        batch_size=args.batch_size, max_length=args.max_length,
    )

    meta_train_df, meta_val_df = train_test_split(
        train_meta, test_size=args.val_split,
        stratify=train_meta["label"], random_state=args.seed,
    )
    print(f"LightGBM train: {len(meta_train_df)}, val: {len(meta_val_df)}")

    print("Training LightGBM...")
    meta_model, best_th = train_lightgbm(meta_train_df, meta_val_df)

    print("Predicting...")
    predictions = predict_lightgbm(meta_model, test_feats, threshold=best_th)
    vals, counts = np.unique(predictions, return_counts=True)
    print(f"  Predictions: {dict(zip(vals, counts))}")

    ids = test_df["id"].values
    write_submission(predictions, output_path=args.output, ids=ids)


def download(args):
    from app.utils.download_data import main as download_main
    import sys
    sys.argv = [sys.argv[0]] + args.extra_args
    download_main()


def main():
    parser = argparse.ArgumentParser(description="Olikbochon — Bengali Hallucination Detector")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--model-name", default="csebuetnlp/banglabert_large")
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-length", type=int, default=256)
    train_parser.add_argument("--skip-xnli", action="store_true")
    train_parser.add_argument("--xnli-epochs", type=int, default=2)
    train_parser.add_argument("--xnli-lr", type=float, default=3e-5)
    train_parser.add_argument("--cv", type=int, default=0, help="Number of CV folds (0 = no CV)")
    train_parser.add_argument("--retrieve", action="store_true", help="Augment context-absent rows with Wikipedia retrieval during training")
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--output", default="submission.csv")
    predict_parser.add_argument("--model-dir", default=None)
    predict_parser.add_argument("--batch-size", type=int, default=16)
    predict_parser.add_argument("--max-length", type=int, default=256)
    predict_parser.add_argument("--retrieve", action="store_true", help="Retrieve Wikipedia passages for context-absent rows")
    predict_parser.set_defaults(func=predict)

    meta_parser = subparsers.add_parser("meta", help="Train meta-classifier via CV OOF probabilities + LightGBM")
    meta_parser.add_argument("--output", default="submission_meta.csv")
    meta_parser.add_argument("--model-name", default="csebuetnlp/banglabert_large")
    meta_parser.add_argument("--batch-size", type=int, default=8)
    meta_parser.add_argument("--max-length", type=int, default=256)
    meta_parser.add_argument("--epochs", type=int, default=10)
    meta_parser.add_argument("--lr", type=float, default=2e-5)
    meta_parser.add_argument("--val-split", type=float, default=0.2)
    meta_parser.add_argument("--seed", type=int, default=42)
    meta_parser.add_argument("--cv", type=int, default=5, help="Number of CV folds for OOF probabilities")
    meta_parser.add_argument("--skip-xnli", action="store_true", help="Skip XNLI warmup")
    meta_parser.add_argument("--xnli-epochs", type=int, default=2)
    meta_parser.add_argument("--xnli-lr", type=float, default=3e-5)
    meta_parser.add_argument("--retrieve", action="store_true", help="Use Wikipedia retrieval for context-absent rows")
    meta_parser.set_defaults(func=meta)

    download_parser = subparsers.add_parser("download", help="Download datasets in Colab")
    download_parser.add_argument("extra_args", nargs="*")
    download_parser.set_defaults(func=download)

    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
