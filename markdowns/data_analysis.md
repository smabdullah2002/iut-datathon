# Dataset & Architecture Analysis

## 1. Dataset Overview

### Samples (`app/dataset/samples.json`) — 299 records

| Property | Value |
|---|---|
| Total records | 299 |
| Faithful (1) | 163 (54.5%) |
| Hallucinated (0) | 136 (45.5%) |
| With context | 132 (44.1%) |
| NULL context | 167 (55.9%) |

### Context × Label Breakdown

| | Faithful | Hallucinated | Total |
|---|---|---|---|
| **Has context** | 84 (63.6%) | 48 (36.4%) | 132 |
| **No context** | 79 (47.3%) | 88 (52.7%) | 167 |

**Key insight**: Hallucinations are more common when no supporting context is provided (52.7% vs 36.4%).

### Response Length Analysis

| Metric | Hallucinated | Faithful |
|---|---|---|
| Avg length | 22 chars | 13 chars |
| Median length | 16 chars | 11 chars |
| Max length | 88 chars | 79 chars |

**Response length distribution**:
| Bucket | Count |
|---|---|
| 1-10 chars | 112 |
| 11-20 chars | 103 |
| 21-50 chars | 70 |
| 51-100 chars | 14 |

**Data quality notes**:
- 2 entries have integer responses (`-2` label=0, `3` label=1) — need casting
- Context `[NULL]` means no supporting context (not missing data)
- Context length when present: 66–2174 chars (avg 608)

### Domains Identified (from prompt keyword heuristics)

| Domain | Approx. Count |
|---|---|
| Geography (rivers, districts, countries) | 51 |
| Bengali grammar/language (idioms, meanings) | 31 |
| Math (algebra, arithmetic) | 21 |
| Science (physics, chemistry, biology) | 21 |
| Literature (poets, novels, writers) | 15 |
| History (liberation war, rulers) | 9 |
| Sports | 6 |
| Law/Constitution | 2 |

### Test Set (`app/dataset/test set.csv`) — 2,516 records

| Property | Value |
|---|---|
| Total records | 2,516 |
| NULL context | 1,155 (45.9%) |
| Has context | 1,361 (54.1%) |
| Label column | Absent (prediction target) |

---

## 2. Qualitative Examples

### Faithful with context
- **Q**: অভ্র কিবোর্ড কে উদ্ভাবন করেন ?
- **A**: মেহদী হাসান খান
- **Ctx**: Supports the fact (context about Avro keyboard inventor)

### Hallucinated with context
- **Q**: তারেক মাসুদ পরিচালিত সর্বশেষ বাংলা চলচ্চিত্রটি কত সালে বাংলাদেশে মুক্তি পায় ?
- **A**: রানওয়ে
- **Ctx**: Says "রানওয়ে" released in 2010, but the question asks for the release *year* of his *last* film — the answer is wrong/contradictory

### Faithful without context
- **Q**: "ধান্ধা" এর ভাবার্থ কী?
- **A**: কোন অসৎ উদ্দেশ্য
- (Common Bengali idiom knowledge)

### Hallucinated without context
- **Q**: 'কাঁঠালপাড়া'য় জন্মগ্রহণ করেন কোন লেখক?
- **A**: শরৎচন্দ্র চট্টোপাধ্যায়
- (Wrong — he was born in Devanandapur, not Kanthalpada)

---

## 3. Architecture Plan Summary

Source: `markdowns/olikbochon_architecture.md`

### High-Level Pipeline

```
Input (prompt, response, context?, C-band)
      │
      ├── Track A: BanglaBERT/XLM-R encoder → faithfulness probability
      ├── Track B: FAISS retrieval from Bengali Wikipedia → top-k similarity/entailment scores
      ├── Track C: TituLLM cross-lingual consistency → compare Bengali vs English answers
      └── Meta-classifier: LightGBM over all features + band-aware calibration
```

### Track Breakdown

| Track | Model | Purpose |
|---|---|---|
| **A** | BanglaBERT / XLM-R (fine-tuned) | Primary faithfulness scorer on `(ctx+resp)` or `(prompt+resp)` |
| **B** | FAISS + sentence-transformer | Retrieval-augmented verification features from Bengali Wikipedia/Banglapedia |
| **C** | TituLLM 1B/3B | Cross-lingual consistency — compare Bengali vs English answer divergence |
| **D** (opt) | TituLLM × 3-5 samples | Self-consistency sampling at moderate temperature |
| **Meta** | LightGBM / Logistic Regression | Final decision over encoder + retrieval + band + context + LLM features |

### Key Strategic Choices

1. **Retrieval feeds features to meta-model, not a standalone branch** — classifier learns when retrieval is informative vs noisy
2. **Cross-lingual consistency is a feature, not a gate** — meta-model learns per-band trust calibration
3. **Band-aware calibration** — C1 is the tie-breaker, model explicitly conditions on cultural-distance bands
4. **Compute budget**: BanglaBERT (tiny), TituLLM 1B (main GPU cost), FAISS (CPU) → fits 9hr / 50GB on P100 or 2×T4

### Tools & Models

| Layer | Tool/Model |
|---|---|
| NLI / entailment | BanglaBERT or XLM-R (fine-tuned) |
| Secondary encoder | XLM-R or BanglaBERT |
| Generative Bengali LLM | TituLLM (1B / 3B) on Llama-3.2 |
| Alternative | TigerLLM |
| Retrieval index | FAISS (local, offline) |
| Sentence embeddings | Bengali/multilingual sentence-transformer |
| Fact corpus | Bengali Wikipedia dump (+ Banglapedia if permitted) |
| Meta-model | LightGBM or Logistic Regression |
| Quantization | GGUF / Q4_K_M if size-constrained |

### Validation Strategy

- Stratified split by C-band (C0/C1/C2) × task type
- Track **overall F1** and **C1 F1 separately**
- Calibrate thresholds per band or include band as categorical feature
- Manual qualitative review of every C1 sample

---

## 4. Architectural Implications from Data

| Observation | Implication |
|---|---|
| ~56% of train and ~46% of test have no context | Track B (retrieval) is critical |
| 72% of responses are ≤20 chars | Model needs fine-grained semantic matching, not long-doc NLI |
| Math/science includes numerical answers | Unique signal path needed |
| Hallu responses are ~70% longer on average | Response length is a weak but usable feature |
| No C-band labels in dataset | Needs manual tagging or heuristic assignment |
| No task-type column | Domains need to be classified |
| No validation split | Needs stratified creation |
| No fact corpus downloaded yet | FAISS index needs to be built |

### Open Items

- C0/C1/C2 band annotation — manual or heuristic?
- Task-type classification schema
- Bengali Wikipedia/Banglapedia corpus sourcing and licensing
- Handling integer responses in preprocessing
- Threshold calibration strategy for hallucinated-class F1 optimization
