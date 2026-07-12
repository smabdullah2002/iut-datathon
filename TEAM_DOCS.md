# অলীকবচন — Project Status & Implementation Notes

## Competition Goal
Build a Bengali hallucination detector. Given (prompt, response, optional context), predict:
- `1` = faithful
- `0` = hallucinated

Scored on **F1 of the hallucinated class**. Tie-breaker: F1 on `C1` subset.

---

## What Has Been Implemented

### Phase 0 — Project Setup & Data Preparation

| Item | Status | Details |
|------|--------|---------|
| Code scaffold | ✅ Done | `app/main.py` entry point, `app/utils/` modules, CLI interface with `train`, `predict`, `download` commands |
| Data loaders | ✅ Done | `load_samples()` reads `samples.json` (299 labeled), `load_test_set()` reads `test set.csv` (2516 unlabeled). Handles integer responses, NULL context flag |
| Preprocessing | ✅ Done | Tokenization with BanglaBERT tokenizer. Input format: `{context} [SEP] {prompt} [SEP] {response}` (or `[SEP] {prompt} [SEP] {response}` when no context) |
| Shared text builder | ✅ Done | `build_input_text(row, sep_token)` — single function used by both train and inference pipelines. Eliminates train/test format drift |

### Phase 1 — Track A: Encoder-Based Classifier

| Item | Status | Details |
|------|--------|---------|
| BanglaBERT fine-tuning | ✅ Done | `AutoModelForSequenceClassification` — binary faithfulness classifier. Training via HuggingFace `Trainer` |
| XNLI warmup | ✅ Done | Uses `csebuetnlp/xnli_bn` (381k Bengali NLI pairs). Maps contradiction→0, entailment→1, discards neutral. Warm-starts the decision boundary before fine-tuning on 299 samples |
| Best checkpoint tracking | ✅ Done | `load_best_model_at_end=True`, metric=`f1_hallucinated`, `greater_is_better=True`. Trainer saves per-epoch checkpoints (only 2 kept via `save_total_limit=2`), reloads the best at end |
| Cross-validation | ✅ Done | `--cv N` flag runs stratified k-fold. Reports per-fold F1 + mean ± std. Uses separate Trainer per fold, no Drive checkpoint bloat |
| Tokenizer drift fix | ✅ Done | `get_tokenizer(model_name=...)` now respects `--model-name`, stays in sync with the encoder |
| Return proba option | ✅ Done | `predict_df(return_proba=True)` returns both class and probability columns for ensemble averaging |

### Phase 2 — Track B: Retrieval-Augmented Features

| Item | Status | Details |
|------|--------|---------|
| Wikipedia download | ✅ Done | `download_data.py:download_wikipedia()` downloads `rejauldu/bengali-wikipedia` (~500k articles), saves as `bn_wiki.txt` |
| FAISS index builder | ✅ Done | `download_data.py:build_faiss_index()` encodes corpus with `paraphrase-multilingual-MiniLM-L12-v2`, builds `bn_wiki.index` (IndexFlatIP) |
| Retrieval module | ✅ Done | `app/utils/retrieval.py` — lazy-loaded singleton. `retrieve_best_passage(prompt, response)` returns top-1 Wikipedia passage. `retrieve_best_passage_with_score(prompt, response)` returns `(passage, score)` tuple |
| Retrieval-augmented inference | ✅ Done | `predict --retrieve` — for context-absent rows, retrieves Wikipedia passage and feeds it as context into BanglaBERT. No retraining needed |
| Retrieval-augmented training | ✅ Done | `train --retrieve` — replaces `[NULL]` context with retrieved passages before training, so the model learns to leverage retrieved evidence |
| Colab notebook cells | ✅ Done | Cells 5a-retrieve (retrieval train), 5b (Track A only), 5c (Track A + B), 5d (meta) in `app/colab_setup.ipynb` |

### Inference Pipeline

| Item | Status | Details |
|------|--------|---------|
| Model loading | ✅ Done | `load_model()` loads from saved dir. Auto-moves to GPU if CUDA available |
| Batched prediction | ✅ Done | `predict_df()` — batch inference with dynamic padding |
| Submission writer | ✅ Done | `write_submission()` — uses real test IDs from `test set.csv`. Fixed from previous 1..N bug |
| GPU support | ✅ Done | Model moved to `.to("cuda")` if available. Inference time: ~20-40s on T4 for 2516 samples |

### Phase 3 — Threshold-Tuned Ensemble

| Item | Status | Details |
|------|--------|---------|
| Multi-seed ensemble | ✅ Done | `threshold --seeds 3` trains BanglaBERT with seeds 42, 43, 44, averages test probabilities |
| OOF threshold search | ✅ Done | Combines val probabilities from all seeds for robust threshold search (~180 samples) |
| No LightGBM | ✅ Done | Direct proba thresholding — simpler, faster, and scores better than stacked meta-classifier |

### Other Fixes

| Item | Status | Details |
|------|--------|---------|
| torchvision/datasets compatibility | ✅ Done | Monkey-patch in `preprocessing.py` for environments where `VideoReader` is missing |

---

## Current Scores

| Configuration | Kaggle F1 (hallucinated) | Notes |
|---------------|--------------------------|-------|
| Track A only (skip-xnli, 10 epochs, lr=2e-5) | 0.523 | Baseline — deprecated, Track A+B is strictly better |
| Track A + Track B (same model, no retrain) | **0.555** | +0.032 just from retrieval at inference |
| Threshold ensemble (3 seeds, skip-xnli, retrieve) | TBD | 3-seed proba averaging + OOF threshold search |

---

## What Has Not Been Implemented Yet

### Phase 0 — C-Band Annotation

| Item | Priority | Effort | Details |
|------|----------|--------|---------|
| Embed → cluster → manual label | **Medium** | ~1 hr | Generate BanglaBERT embeddings for 299 samples → K-Means cluster → manually assign C0/C1/C2. Needed for tie-breaker and band-aware calibration |
| Band-aware threshold tuning | **Medium** | 1 hr | Search optimal decision threshold per C-band after annotation |

### Phase 3 — Track C: Cross-Lingual Consistency

| Item | Priority | Effort | Details |
|------|----------|--------|---------|
| TituLLM 1B setup + quantization | **Medium** | 2-3 hrs | Download, optionally quantize with GGUF/Q4_K_M. Must fit 50GB weight limit |
| Claim extraction | **Medium** | 1 hr | Extract core factual claim from Bengali response (heuristic: remove filler) |
| Dual-language query | **Medium** | 2 hrs | Ask TituLLM in Bengali and English, compare answer similarity |
| Consistency scoring | **Medium** | 1 hr | Compute semantic similarity / entailment between Bengali and English answers |
| Integration as feature | **Low** | 1 hr | Feed consistency score into meta-classifier |

### Phase 4 — Ensemble

| Item | Priority | Effort | Details |
|------|----------|--------|---------|
| Multi-seed ensemble training | **High** | 3× train runs | Train 3-5 runs with different seeds (42, 43, 44), average probabilities at inference. Usually +0.02-0.04 F1 |
| Ensemble averaging script | **Medium** | ~30 min | Read multiple `predict_df` probability outputs, average, write submission |

### Phase 5 — Documentation & Package

| Item | Priority | Effort | Details |
|------|----------|--------|---------|
| 4-page paper | **Low** | 4-6 hrs | Problem, approach, experiments, results |
| README | **Low** | 1-2 hrs | Environment, model details, reproduction |
| Kaggle notebook package | **Low** | 2-3 hrs | Runnable offline notebook with pre-downloaded assets |

---

## Recommended Order to Continue

| Step | What | Est. Time | Expected Gain |
|------|------|-----------|---------------|
| 1 | Threshold ensemble: 3 seeds + retrieval + OOF threshold | 1 GPU run (~30 min) | +0.01–0.04 |
| 2 | XNLI warmup + threshold ensemble | 1 long GPU run | +0.05–0.10 combined |
| 3 | C-band annotation + band-aware threshold tuning | ~1 hr manual | +0.01–0.02 |
| 4 | TituLLM cross-lingual consistency (Track C) | ~4 hrs | +0.02–0.05 |
| 5 | Documentation + Kaggle package | ~6 hrs | Required for Phase 2 |

---

## How to Run

### Clone & setup
```bash
git clone https://github.com/smabdullah2002/iut-datathon.git
cd iut-datathon
pip install -r requirements.txt
```

### Train
```bash
# Fast baseline (skip XNLI)
python -m app.main train --skip-xnli --epochs 10 --batch-size 16 --lr 2e-5

# With XNLI warmup (better, but much slower)
python -m app.main train --epochs 10 --batch-size 16 --lr 2e-5

# With retrieval augmentation (best so far)
python -m app.main train --skip-xnli --epochs 10 --batch-size 16 --lr 2e-5 --retrieve

# With cross-validation
python -m app.main train --cv 5 --epochs 10 --batch-size 16 --lr 2e-5
```

### Predict
```bash
# Track A only
python -m app.main predict --output submission.csv

# Track A + Track B (with retrieval)
python -m app.main predict --output submission.csv --retrieve
```

### Threshold Ensemble (recommended)
```bash
# 3-seed ensemble with OOF threshold search (fast, ~30 min on T4)
python -m app.main threshold --skip-xnli --retrieve --epochs 10 --batch-size 16 --seeds 3

# With XNLI warmup (better, ~3-4 hrs on T4)
python -m app.main threshold --retrieve --epochs 10 --batch-size 16 --seeds 3
```

### Download datasets + build FAISS index
```bash
python -m app.main download
```
