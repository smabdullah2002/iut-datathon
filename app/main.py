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
    import torch
    from app.utils.preprocessing import (
        load_samples, split_samples, load_test_set, get_tokenizer,
        load_xnli_bn, make_xnli_dataset, make_samples_dataset, set_seed,
    )
    from app.utils.inference import write_submission
    from app.utils.meta import oof_meta_features, extract_features, train_lightgbm, predict_lightgbm
    from app.utils.training import get_model, train_xnli_warmup, train_samples
    from sklearn.model_selection import train_test_split

    print("Loading datasets...")
    train_df = load_samples()
    test_df = load_test_set()
    has_bands = "band" in train_df.columns

    tokenizer = get_tokenizer(model_name=args.model_name)
    use_retrieval = args.retrieve
    if use_retrieval:
        print("Using retrieval for context-absent rows")

    seeds = [42, 43, 44][:max(1, args.seeds)]
    n_seeds = len(seeds)

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

    base_model_state = base_model.state_dict()

    n_splits = max(2, args.cv)
    train_meta_list = []
    test_feats_list = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{n_seeds})")
        print(f"{'='*60}")
        set_seed(seed)

        print(f"Generating OOF probabilities ({n_splits}-fold CV)...")
        train_meta = oof_meta_features(
            train_df, tokenizer, n_splits=n_splits,
            model_name=args.model_name,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            max_length=args.max_length, seed=seed,
            use_retrieval=use_retrieval,
            xnli_model=base_model if not args.skip_xnli else None,
        )

        print("Training final model with early stopping...")
        final_split_train, final_split_val = split_samples(train_df, val_size=args.val_split, random_state=seed)
        final_train_ds = make_samples_dataset(tokenizer, final_split_train, max_length=args.max_length)
        final_val_ds = make_samples_dataset(tokenizer, final_split_val, max_length=args.max_length)

        final_model = get_model(num_labels=2, model_name=args.model_name)
        if not args.skip_xnli:
            final_model.load_state_dict(base_model_state)
        final_model, _ = train_samples(
            final_model, tokenizer, final_train_ds, final_val_ds,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            save_checkpoints=False,
        )

        print("Extracting test features...")
        test_feats = extract_features(
            final_model, tokenizer, test_df,
            use_retrieval=use_retrieval,
            batch_size=args.batch_size, max_length=args.max_length,
        )

        train_meta_list.append(train_meta)
        test_feats_list.append(test_feats)
        del final_model
        torch.cuda.empty_cache()

    del base_model, base_model_state
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Averaging across {n_seeds} seeds...")
    print(f"{'='*60}")

    if n_seeds > 1:
        avg_meta = train_meta_list[0].copy()
        avg_meta["proba_1"] = np.mean([df["proba_1"].values for df in train_meta_list], axis=0)

        avg_test = test_feats_list[0].copy()
        avg_test["proba_1"] = np.mean([df["proba_1"].values for df in test_feats_list], axis=0)
    else:
        avg_meta = train_meta_list[0]
        avg_test = test_feats_list[0]

    meta_train_df, meta_val_df = train_test_split(
        avg_meta, test_size=0.3,
        stratify=avg_meta["label"], random_state=42,
    )
    print(f"LightGBM train: {len(meta_train_df)}, val: {len(meta_val_df)}")

    print("Training LightGBM with band-aware thresholding...")
    meta_model, best_th, band_thresholds = train_lightgbm(meta_train_df, meta_val_df)
    print(f"  Using global threshold: {best_th:.3f}")

    print("Predicting...")
    predictions = predict_lightgbm(meta_model, avg_test, threshold=best_th)
    vals, counts = np.unique(predictions, return_counts=True)
    print(f"  Predictions: {dict(zip(vals, counts))}")

    ids = test_df["id"].values
    write_submission(predictions, output_path=args.output, ids=ids)


def annotate(args):
    from app.utils.annotate import main as annotate_main
    annotate_main(args)


def threshold(args):
    import numpy as np
    import torch
    from app.utils.preprocessing import (
        load_samples, split_samples, load_test_set, get_tokenizer,
        load_xnli_bn, make_xnli_dataset, make_samples_dataset, set_seed,
    )
    from app.utils.inference import predict_df, write_submission
    from app.utils.meta import search_threshold
    from app.utils.training import get_model, train_xnli_warmup, train_samples, save_model

    seeds = [42, 43, 44][:max(1, args.seeds)]
    n_seeds = len(seeds)
    all_test_proba_1 = None

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{n_seeds})")
        print(f"{'='*60}")
        set_seed(seed)

        print("Loading datasets...")
        train_df = load_samples()
        test_df = load_test_set()

        tokenizer = get_tokenizer(model_name=args.model_name)

        if args.retrieve:
            print("Augmenting context-absent rows with Wikipedia retrieval...")
            from app.utils.retrieval import retrieve_best_passage
            missing = train_df["context"] == "[NULL]"
            retrieved = [retrieve_best_passage(p, r) for p, r in zip(train_df.loc[missing, "prompt_bn"], train_df.loc[missing, "response_bn"])]
            train_df.loc[missing, "context"] = retrieved
            filled = (train_df.loc[missing, "context"] != "").sum()
            print(f"  Retrieved {filled}/{missing.sum()} passages")

        train_split, val_split = split_samples(train_df, val_size=args.val_split, random_state=seed)
        print(f"Train: {len(train_split)}, Val: {len(val_split)}")

        model = get_model(num_labels=2, model_name=args.model_name)

        if not args.skip_xnli:
            print("Loading xnli_bn for warm-up...")
            xnli_df = load_xnli_bn()
            print(f"  {len(xnli_df)} NLI pairs loaded")
            xnli_dataset = make_xnli_dataset(tokenizer, xnli_df, max_length=args.max_length)
            print(f"  Warm-up training for {args.xnli_epochs} epochs...")
            model = train_xnli_warmup(
                model, tokenizer, xnli_dataset,
                batch_size=args.batch_size, epochs=args.xnli_epochs, lr=args.xnli_lr,
            )
            print("  Warm-up complete.")

        train_dataset = make_samples_dataset(tokenizer, train_split, max_length=args.max_length)
        val_dataset = make_samples_dataset(tokenizer, val_split, max_length=args.max_length)

        print(f"Fine-tuning for {args.epochs} epochs...")
        model, _ = train_samples(
            model, tokenizer, train_dataset, val_dataset,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            save_checkpoints=False,
        )

        print("Getting val probabilities...")
        val_proba = predict_df(model, tokenizer, val_split, batch_size=args.batch_size,
                               max_length=args.max_length, use_retrieval=args.retrieve, return_proba=True)
        val_proba_1 = val_proba[:, 1]
        val_labels = val_split["label"].values

        print("Searching best threshold...")
        best_th = search_threshold(val_proba_1, val_labels)

        print("Running test inference...")
        test_proba = predict_df(model, tokenizer, test_df, batch_size=args.batch_size,
                                max_length=args.max_length, use_retrieval=args.retrieve, return_proba=True)
        test_proba_1 = test_proba[:, 1]

        if all_test_proba_1 is None:
            all_test_proba_1 = test_proba_1
        else:
            all_test_proba_1 += test_proba_1

        del model
        torch.cuda.empty_cache()

    all_test_proba_1 /= n_seeds

    predictions = (all_test_proba_1 >= best_th).astype(int)
    vals, counts = np.unique(predictions, return_counts=True)
    print(f"\nPredictions: {dict(zip(vals, counts))}")
    print(f"Threshold: {best_th:.3f}")

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
    meta_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for ensemble (default 3)")
    meta_parser.set_defaults(func=meta)

    annotate_parser = subparsers.add_parser("annotate", help="Generate C-band annotation template")
    annotate_parser.add_argument("--clusters", type=int, default=10, help="Number of K-Means clusters")
    annotate_parser.add_argument("--output", default="cband_annotation.csv", help="Output CSV path")
    annotate_parser.add_argument("--auto", action="store_true", help="Auto-fill bands using keyword heuristics per cluster")
    annotate_parser.set_defaults(func=annotate)

    threshold_parser = subparsers.add_parser("threshold", help="Train encoder + optimize threshold directly (no LightGBM)")
    threshold_parser.add_argument("--output", default="submission_threshold.csv")
    threshold_parser.add_argument("--model-name", default="csebuetnlp/banglabert_large")
    threshold_parser.add_argument("--batch-size", type=int, default=16)
    threshold_parser.add_argument("--max-length", type=int, default=256)
    threshold_parser.add_argument("--epochs", type=int, default=30)
    threshold_parser.add_argument("--lr", type=float, default=2e-5)
    threshold_parser.add_argument("--val-split", type=float, default=0.2)
    threshold_parser.add_argument("--seed", type=int, default=42)
    threshold_parser.add_argument("--skip-xnli", action="store_true")
    threshold_parser.add_argument("--xnli-epochs", type=int, default=2)
    threshold_parser.add_argument("--xnli-lr", type=float, default=3e-5)
    threshold_parser.add_argument("--retrieve", action="store_true")
    threshold_parser.add_argument("--seeds", type=int, default=1, help="Number of seeds for ensemble")
    threshold_parser.set_defaults(func=threshold)

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
