import os
import sys
import argparse

COLAB = "COLAB_GPU" in os.environ

if COLAB:
    BASE_DIR = "/content/iut-datathon"
    DATA_DIR = "/content/dataset"
    MODELS_DIR = "/content/drive/MyDrive/iut_datathon_models"
    CORPUS_DIR = "/content/corpus"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
    CORPUS_DIR = os.path.join(BASE_DIR, "..", "corpus")


def train(args):
    """Fine-tune BanglaBERT on the hallucination detection task."""
    print("Training mode")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  Corpus dir: {CORPUS_DIR}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print("Training not yet implemented.")


def predict(args):
    """Run inference on the test set and produce submission CSV."""
    print("Inference mode")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Output: {args.output}")
    print("Inference not yet implemented.")


def download(args):
    """Download datasets and build retrieval index in Colab."""
    from app.utils.download_data import main as download_main
    sys.argv = [sys.argv[0]] + args.extra_args
    download_main()


def main():
    parser = argparse.ArgumentParser(description="Olikbochon — Bengali Hallucination Detector")
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--output", default="submission.csv")
    predict_parser.set_defaults(func=predict)

    download_parser = subparsers.add_parser("download", help="Download datasets in Colab")
    download_parser.add_argument("extra_args", nargs="*", help="Arguments passed to download_data.py")
    download_parser.set_defaults(func=download)

    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
