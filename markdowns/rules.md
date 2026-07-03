# Simplified Rules

## 1. Team Rules

- Teams may have 1 to 4 members.
- A participant may belong to only one team.
- Team mergers are allowed only before the Phase 1 merger deadline.
- At least one teammate must remain reachable by email.

## 2. Competition Structure

### Phase 1
- Submit CSV predictions to Kaggle.
- The leaderboard is based on separate public and private test splits.

### Phase 2
- Top teams submit a complete solution package.
- Organizers run the package on a hidden held-out fold.
- Final ranking is based on a weighted combination of leaderboard score, held-out score, presentation, paper, and novelty.

## 3. Submission Requirements

### Phase 1 submission
- File format: CSV
- Columns: `id`, `label`
- `label` must be `0` or `1`
- No missing rows, extra columns, or invalid values
- Up to 4 submissions per day per team

### Phase 2 package
- Runnable Kaggle notebook
- 4-page paper PDF
- Short README with environment and model details

## 4. Compute and Inference Rules

- Only open-weight models may be used at inference
- No paid APIs or closed services
- Inference must run offline in Kaggle
- Total runtime must be under 9 hours
- Total model size must be under 50 GB

## 5. Data and Model Use

- Public Bengali or multilingual data may be used
- Open-weight models are allowed
- Teams may use their own trained or fine-tuned models
- External data augmentation is allowed
- Fine-tuning on the released sample set is allowed
- Test labels may not be used

## 6. Fair Play

- Do not share competition code privately across teams
- Do not probe the leaderboard to recover hidden labels
- Do not misrepresent authorship
- Report suspected dataset issues publicly

## 7. Scoring

- Primary metric: binary F1 on the hallucinated class
- Phase 1 tie-breaker: higher F1 on the `C1` subset
- If still tied, earlier qualifying submission wins

## 8. Eligibility

- Contestants must be currently enrolled undergraduate students from a recognized university in Bangladesh
- Travel to the final is expected for top teams, with remote participation available when needed

## 9. Notes

These are simplified working notes. For the exact competition terms, read the official Kaggle rules and timeline.