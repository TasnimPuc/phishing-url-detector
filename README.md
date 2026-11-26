# PhishGuard URL: real-time phishing detection from the raw URL

A character-level CNN reads **only the URL string**, with no HTML and no page features. It is small (~90k parameters, 360 KB) and runs
**inside the browser** as a Chrome extension, so no URL is ever sent anywhere.

```
raw URL ─► normalise ─► char ids ─► Char-CNN ─► p(phishing) ─► badge / warning page
```

## Why this repo is different
- **Raw string only.** The dataset also ships many derived features (`URLSimilarityIndex`, HTML counts…). Using them gives
  near-perfect but unrealistic scores. Here they are ignored.
- **Honest evaluation.** URLs are normalised (scheme, `www.` and trailing `/` removed), de-duplicated, and split **by registered
  domain**, so the test set contains only domains the model has never seen.
- **Shortcut audit + stress test.** In PhiUSIIL the legitimate URLs are mostly bare homepages, so "has a path" alone predicts
  phishing. The notebook measures this and adds paths such as `/login` and `/search?q=…` to legitimate test URLs. A model that
  learned the shortcut raises false alarms on normal browsing. The extension ships the model that survives this test (by default
  the hostname model).
- **Same code in Python and JS.** `extension/model.js` repeats the notebook's normalisation, encoding and forward pass;
  `tests/test_parity.py` checks they agree (< 1e-5).

## Results
After a run, see `results/RESULTS.md` (shortcut audit, test metrics with 95% CI, stress test).
Thresholds come from validation legitimate URLs: **warn** = 1% false-positive rate, **block** = 0.1%.

## Run (Kaggle)
1. Import `notebooks/phishing_url_detector.ipynb` → Internet **On** → GPU T4 (optional) → *Run All* (≈ 15–25 min).
2. Optional: Add-ons → Google Cloud Services → Cloud Storage. Each epoch is checkpointed to the bucket, so a stopped session
   resumes. Set `project_id` / `bucket` in the Config cell.
3. Download from the Output tab: `outputs/models/`, `outputs/results/`, `outputs/extension/`.

## Install the extension
1. Copy `outputs/extension/` from the run into `extension/` here (adds `weights.js` and `icons/`).
2. Chrome → `chrome://extensions` → **Developer mode** → **Load unpacked** → choose the `extension/` folder.
3. Browse. A badge shows `!` (phishing), `?` (suspicious) or nothing (safe). Pages above the block threshold open a warning
   page with *Go back* / *Proceed anyway*. The popup shows the score and lets you trust a site.

## Repository
```
notebooks/phishing_url_detector.ipynb   full pipeline (data → audit → models → export → extension)
extension/                              Chrome MV3 extension (pure JS inference, no network)
models/                                 charcnn_host / charcnn_full  (.pth + .onnx)   ← from the run
results/                                RESULTS.md, results.json, training_curves.png ← from the run
tests/test_parity.py                    JS ↔ PyTorch parity
tools/build_notebook.py                 regenerates the notebook from extension/ sources
```

## Data
UCI Machine Learning Repository, **PhiUSIIL Phishing URL Dataset** (id 967): 134,850 legitimate + 100,945 phishing URLs.
Original label: 1 = legitimate, 0 = phishing (this repo uses `y = 1` for phishing).
Prasad, A. & Chandra, S. (2024). *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on
similarity index and incremental learning.* Computers & Security, 136, 103545.

## Limits
A URL-only model cannot see the page, so phishing on trusted hosting (e.g. a Google Sites page) is hard to catch from the
hostname alone. Use it as one layer next to the browser's built-in protection, not as a replacement.

## License
Code: MIT.
