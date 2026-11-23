# Model card: `night_4.h5`

| | |
|---|---|
| Model | Dense autoencoder 52-256-128-64-32-64-128-256-52, 113,620 params (Keras, HDF5) |
| Version | Written and trained in 2025. Frozen. Re-packaged November 2026 |
| Task | Unsupervised anomaly scoring of bank-style user profiles |
| Output | Reconstruction MSE (higher means more anomalous). Alert if > 0.0578 (`models/threshold.json`) |
| Companion artifacts | `models/scaler.pkl` (original MinMaxScaler), `models/label_encoders.json` (recovered and verified vocabularies) |

## Intended use
- Rank or flag user accounts **for human review**.
- It is a triage tool, not an automated decision-maker. It must not be used on its own to block accounts or deny service.
- It is a portfolio and educational artifact: it was trained on synthetic data only.

## Training data
100,000 synthetic "normal" users (80,000 used for fitting), generated with Faker and NumPy. All features are independent uniform draws. There are no real people in the data. See `docs/REPORT.md` §2-3.

## Evaluation (held-out, threshold fixed on separate calibration normals)
| Metric | Value |
|---|---|
| Recall, 2,500 synthetic fraud users | 100% |
| FPR, 10,000 held-out normal users | 1.04% |
| Precision at an assumed 1% prevalence | 49% |
| ROC-AUC / AP | 1.000 / 1.000 (an IsolationForest baseline also scores 0.99999) |

## Limitations
- **Not validated on real data.** The synthetic fraud is separable by construction, so these numbers are an upper bound.
- The input includes leaky and circular features (suspension/report counters, pre-computed risk scores). The probes show recall does not depend on them.
- Label encoding means novel categorical values carry little weight.
- It flags *unusual* users, which includes legitimate edge cases.
- The input schema is fixed: exactly the 52 named columns. Missing columns or non-numeric values are rejected.

## Ethical considerations
- `age`, `gender` and location are model inputs. They carry no signal in this data (identically distributed in both populations), but on real data they could create disparate alert rates.
- A real deployment would need per-segment FPR monitoring, or these inputs removed.

## Security
HDF5 models and pickled scalers can execute code when loaded. Load them only from this repository or another trusted source.
