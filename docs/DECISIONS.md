# Decision log (ADRs)

Each entry records one decision: its context, the alternatives considered, and the consequence. Dates are when the decision was made.

---

### ADR-001: Rebuild around the original weights; do not retrain (2026-11-23)
**Context.** This is a 2025 hackathon project. The trained `night_4.h5` and its `scaler.pkl` survived. The surrounding code was broken.
**Decision.** Keep the 2025 weights frozen and re-engineer everything around them: preprocessing, threshold, evaluation, serving and docs.
**Alternatives.** Retrain with a better feature set. Rejected for this release: it would erase the provenance of the shipped model, and the data is too easy to show any improvement (see REPORT §7).
**Consequence.** The input contract (52 features, label encoding, MinMax) is fixed. Feature-engineering improvements are documented as v2 recommendations, not applied.

### ADR-002: Drop identifier columns
**Decision.** Drop `user_id`, `account_number`, `transaction_id` and `Unnamed: 0` before encoding. `user_id` is kept separately to label the output.
**Why.** They are random identifiers with no behavioural meaning. Keeping them would let a model memorise rows.

### ADR-003: Recover the lost label encoders deterministically and verify against the scaler
**Context.** `label_encoders.pkl` was written by the training notebook but is missing.
**Decision.** Refit the vocabularies (sorted unique values, which is what `LabelEncoder` does) on the exact training CSV. Accept them only if encoding the CSV reproduces the per-feature min/max stored in the original `scaler.pkl`.
**Result.** All 52 of 52 features match, so the recovery is exact. Stored as JSON rather than pickle: readable, diffable, and carries no code-execution risk.

### ADR-004: Unseen categories map to −1
**Alternatives.**
- Raise an error: breaks scoring for any new category.
- Map to the most frequent class: hides novelty.
- Map to −1: the hackathon's own `encode_with_fallback` convention.

**Decision.** Use −1. It lands just outside the training range, in the anomalous direction, without breaking the request.
**Known weakness.** The signal is weak: an unseen `email_domain` alone almost never triggers an alert (REPORT §7). The real fix is one-hot encoding with an `is_unseen` flag, which needs retraining. Meanwhile the API returns `unseen_category_rate` per request so drift is visible.

### ADR-005: Threshold = 99th percentile of calibration-normal scores (target FPR 1%)
**Context.** The hackathon used three rules:
- a fixed 0.05 (`third.ipynb`);
- the batch 96th percentile (`api.py`);
- Tukey's fence (`third(1).ipynb`).

**Decision.** Use a quantile of anomaly scores on 10k held-out normal users that the model never trained on, with a target FPR of 1%. Stored with its provenance in `models/threshold.json`.
**Why.**
- Every candidate rule reaches 100% recall on this data, so the only thing that differs is alert volume.
- The quantile rule makes alert volume an explicit business parameter (analyst capacity).
- It is also well calibrated: 1.04% FPR on unseen test normals.

**Rejected.**
- Batch percentile: flags k% of every upload regardless of content.
- Fixed 0.05: 4.3% FPR, with no principled origin.
- MAD and skew rules: 4.5-6.8% FPR.

### ADR-006: Preprocessing is fitted once and applied only; scoring is row-independent
**Context.** The hackathon API re-fitted `LabelEncoder` and `MinMaxScaler` on every uploaded CSV. Uploading the 2,500-fraudster demo file, it flagged only 4% of them, because after per-batch scaling the fraudsters look normal relative to each other.
**Decision.** Load the frozen artifacts and only call `transform`. A unit test asserts that a row's vector is the same whether it is scored alone or inside a batch.
**Other API changes.**
- Schema validation, with 422 errors that list the missing columns.
- Upload size limit.
- Per-user `top_features` explanation.
- `/health` endpoint.
- Serves `night_4.h5` (the old API loaded an older, weaker checkpoint).

### ADR-007: Evaluation split and metrics
**Decision.**
- Reproduce the training split (`random_state=42`) so the evaluation normals are exactly the model's validation rows.
- Split those 50/50 into calibration and test.
- Use fraud rows only for reporting.
- Report recall at a fixed FPR and precision at an assumed 1% prevalence, alongside AUC.

**Why.** It is leak-free by construction. AUC alone is uninformative on a saturated benchmark, and raw precision on a 20%-fraud test set would overstate real-world precision by about 25×.

### ADR-008: Keep the pre-computed `*_score` and suspension/report counters in the input, but prove recall does not depend on them
**Context.** These features are circular (other models' outputs) or post-hoc labels. They cannot be removed without retraining.
**Decision.** Keep them (frozen contract), and add a counterfactual probe that replaces them with normal values. Recall stays at 100%, so the model's decisions do not hinge on them. Removal is listed first in the v2 feature set.

### ADR-009: Exclude the team's Django app from the public repo
**Decision.** Do not publish the Django staff portal.
**Why.**
- It contains a hard-coded Django `SECRET_KEY`, a shared plaintext seed password and a SQLite database with password hashes and session keys.
- Its prediction endpoint was not routed and its model files were missing.
- It is a separate supervised model on a separate dataset (PaySim), not part of this pipeline.

### ADR-010: Commit the original training data (gzipped)
**Decision.** Commit `non_fraud_night_1.csv.gz` (24 MB; 52 MB uncompressed).
**Why.** The model, scaler and recovered encoders are only verifiable against this exact file. Regenerating it would produce different rows, because the original run was unseeded. All data is synthetic, so there is no privacy concern.
