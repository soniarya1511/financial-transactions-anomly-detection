# Legacy hackathon code (verbatim, 2025)

Kept unchanged for provenance. **Do not use**: see `docs/REPORT.md` §10 for what was wrong and how it was fixed.

| File | What it was |
|---|---|
| `train_autoencoder_original.ipynb` | `third.ipynb`: data generation, training of `night_4.h5`, fixed-0.05 threshold scoring |
| `threshold_experiments_original.ipynb` | `third(1).ipynb`: same training cell plus MAD / Tukey / skew / hybrid threshold experiments |
| `api_original.py` | The hackathon FastAPI service (crashes: missing imports, undefined names, per-batch refitting) |
| `ideas.txt` | Initial brainstorm (k-means → autoencoder) |
