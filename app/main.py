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
    train_df, val_df = split_samples(df, val_size=args.val_split, random_state=args.seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    tokenizer = get_tokenizer()
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
    model = train_samples(
        model, tokenizer, train_dataset, val_dataset,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
    )

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

    print("Running inference...")
    predictions = predict_df(model, tokenizer, test_df, batch_size=args.batch_size, max_length=args.max_length)

    vals, counts = np.unique(predictions, return_counts=True)
    label_dist = dict(zip(vals, counts))
    print(f"  Predictions: {label_dist}")

    write_submission(predictions, output_path=args.output)


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
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--output", default="submission.csv")
    predict_parser.add_argument("--model-dir", default=None)
    predict_parser.add_argument("--batch-size", type=int, default=16)
    predict_parser.add_argument("--max-length", type=int, default=256)
    predict_parser.set_defaults(func=predict)

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
