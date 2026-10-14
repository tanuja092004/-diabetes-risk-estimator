# Diabetes Risk Estimator

A logistic-regression project built from `Python_Implementation.ipynb` and `diabetes1.csv`
(Pima Indians Diabetes dataset, 768 rows). It has two parts:

1. **Training** (`train_model.py`) reproduces the notebook pipeline exactly:
   zeros -> column median, min-max scaling (all columns except Pregnancies/Outcome),
   80/20 split with `random_state=45`, `LogisticRegression()`.
   It writes `model.json` (weights, scaler ranges, medians, metrics, chart data).
2. **Web app** (`index.html`) is one static file. The model runs in the browser in
   plain JavaScript, so there is no server and no data leaves the user's device.

Test-set results: accuracy 75.3%, precision 65.0%, recall 52.0%, F1 57.8%, AUC 0.835.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py      # retrain, writes model.json
python build_site.py       # inlines model.json into index.html
python -m http.server 8000 # open http://localhost:8000
```
(You can also just double-click `index.html`.)

## Deploy (any static host, no build step needed)
- **Netlify:** drag the project folder onto https://app.netlify.com/drop
- **GitHub Pages:** push the folder to a repo, then Settings -> Pages -> Deploy from branch (root).
- **Vercel:** `npx vercel` in this folder, accept the defaults (framework: Other).

## Files
| File | Purpose |
|---|---|
| `index.html` | Finished web app (model embedded) |
| `index.template.html` | Same app with a placeholder for the model |
| `train_model.py` | Trains the model, exports `model.json` |
| `build_site.py` | Builds `index.html` from the template + `model.json` |
| `data/diabetes1.csv` | Dataset |
| `notebook_original.ipynb` | Your original notebook |

## Notes
Educational project, not a medical device. As in the original notebook, medians
and the scaler are computed on the full dataset before the split, which slightly
flatters test scores; fit them on the training split only for a stricter evaluation.
