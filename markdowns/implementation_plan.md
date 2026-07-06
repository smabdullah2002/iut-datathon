# Implementation Plan — অলীকবচন Hallucination Detector

## Phase 0: Project Setup & Data Preparation

- [ ] Scaffold the codebase: Populate `app/main.py` as the entry point, create `app/utils/preprocessing.py`, `app/utils/evaluation.py`, `app/utils/retrieval.py`
- [ ] Build data loader: Read `samples.json` and `test set.csv`, handle integer responses (cast to str), create stratified train/val split by context-availability and domain
- [ ] Preprocessing pipeline: Tokenize Bengali text, align `(context, prompt, response)` into model input format
- [ ] C-band annotation (semi-automatic):
  1. Generate semantic embeddings for all 299 samples using BanglaBERT
  2. Cluster embeddings with K-Means to group semantically similar samples
  3. Present clusters for manual review
  4. Manually assign C-band labels, correcting any cluster misfits
  5. Note: clustering only organizes samples; human annotation is the ground truth

## Phase 1: Track A — Encoder-Based Classifier

- [ ] Fine-tune **BanglaBERT** as binary faithfulness classifier on the 299 samples. Input = `[CLS] context/prompt [SEP] response [SEP]`. Output = probability of faithful.
- [ ] Cross-validation: Stratified 5-fold by context-availability, report hallucinated-class F1
- [ ] **Data augmentation**: Use `csebuetnlp/xnli_bn` (381k Bengali NLI pairs on HuggingFace) — map contradiction→0, entailment→1, discard neutral — to warm-start the decision boundary. Use BNLI as a clean evaluation set.
- [ ] Track both sub-modes: Separate fine-tuning heads or input formatting for context-present vs context-absent rows

## Phase 2: Track B — Retrieval-Augmented Features

- [ ] Download & prepare corpus: **Bengali Wikipedia dump** as the fact corpus
- [ ] Build FAISS index: Encode corpus with Bengali sentence-transformer → create dense index
- [ ] Retrieval pipeline: For each `(prompt, response)` pair, retrieve top-k passages from the corpus
- [ ] Feature extraction:
  - Top-k similarity scores (cosine distance)
  - Best-match entailment/contradiction score between retrieved passage and response
  - Retrieval coverage (how much of the response is covered by retrieved text)
  - Evidence confidence (consistency across top-k results)
- [ ] Save as enrichment features on train/val sets

## Phase 3: Track C — Cross-Lingual Consistency

- [ ] Set up **TituLLM 1B**: Download and quantize with GGUF if needed
- [ ] Claim extraction: Extract the core factual claim from the Bengali response (simple heuristic: remove filler phrases)
- [ ] Dual-language query: Ask TituLLM the same question in Bengali and English (using prompt templates)
- [ ] Consistency scoring: Compute semantic similarity between Bengali and English answers (sentence-transformer cosine), or small entailment check
- [ ] Large divergence → strong hallucination signal: Feed as feature to meta-model

## Phase 4: Meta-Classifier

- [ ] Feature assembly: Stack Track A logits + Track B retrieval features + Track C consistency score + metadata (context flag, C-band, domain, response length)
- [ ] Train meta-model: LightGBM with hallucinated-class F1 optimization, stratified validation by band
- [ ] Threshold calibration: Search optimal decision threshold per C-band to maximize hallucinated-class F1
- [ ] Ablation study: Measure contribution of each feature track to overall and C1 F1

## Phase 5: Inference Pipeline & Submission

- [ ] Build `app/main.py` inference pipeline: Load all models, process test set, output `id,label` CSV
- [ ] Kaggle notebook wrapper: Create a single Kaggle notebook that runs the entire pipeline offline
- [ ] Validate on test set: Ensure < 9hr runtime, < 50GB weights
- [ ] Ensemble (optional): If time allows, ensemble 3-5 fine-tuned encoder runs with different seeds
- [ ] Submission: Format as `submission.csv` with `id,label` columns

## Phase 6: Documentation & Phase 2 Package

- [ ] 4-page paper: Problem statement, approach, experiments, results
- [ ] README: Environment, model details, reproduction steps
- [ ] Kaggle package: Runnable notebook with all dependencies and pre-downloaded models

---

## Confirmed Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Encoder** | BanglaBERT | Bengali-specific, smaller/faster than XLM-R, same team behind xnli_bn dataset |
| **C-band annotation** | Semi-automatic (embed → cluster → human review) | Clusters organize similar samples to speed up manual labeling; human is ground truth |
| **Retrieval corpus** | Bengali Wikipedia dump | Largest open Bengali factual corpus, easy to index with FAISS |
| **Cross-lingual model** | TituLLM 1B | Fits 9hr / 50GB budget; 3B if budget allows |
| **Data augmentation** | `csebuetnlp/xnli_bn` (primary) + BNLI (eval) | Both are public Bengali data, explicitly permitted under rules; contradiction→0, entailment→1, discard neutral |
