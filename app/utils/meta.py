import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def search_threshold(probs, y_true):
    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.05, 0.95, 0.01):
        preds = (probs >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
        if f1[0] > best_f1:
            best_f1 = f1[0]
            best_th = th
    print(f"  Best threshold: {best_th:.3f} (f1_hallucinated: {best_f1:.4f})")
    return best_th
