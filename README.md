# EA-IT2-UGF: explanation-audited interval type-2 fuzzy UGF on the Kelmarsh wind farm

Code, derived data and outputs for the article

> Bhardwaj L, Sharma MK. *Does Explanation-Based Screening Improve LLM-Based Fuzzy Forecasts of Wind-Farm Availability? A Controlled Test* (manuscript, 2026).

The study asks whether screening large language model (LLM) health ratings of wind turbines by their explanations improves
next-month availability forecasts beyond unscreened ratings and beyond random removal of the same number of ratings. An LLM
(Phi-3.5-mini) rates each turbine from monthly condition reports in three wordings, exact Shapley values measure how much of each
rating comes from irrelevant sentences, an audit gate removes the worst wordings, and the retained ratings form interval type-2 (IT2)
fuzzy health numbers that a universal generating function (UGF) propagates exactly into availability bands. On eight years of
Kelmarsh data the gate narrowed the bands by 10 % but did not beat matched random removal.

## Contents

| Path | What it is |
|---|---|
| `xfmss/` | Python package: UGF (`ugf.py`), IT2 numbers and propagation (`it2.py`), LLM reading of letter probabilities (`llm.py`), explainers (`explain.py`), metrics (`metrics.py`), report assembly (`benchmark.py`) |
| `tools/download_data.py` | Downloads the Kelmarsh data from Zenodo (MD5-checked) |
| `tools/extract_kelmarsh.py` | Raw SCADA and status files -> 10-minute tables (`--src`, `--out`) |
| `tools/build_kelmarsh.py` | Operating states, state shares, condition reports -> `data/kelmarsh.json`, `data/real/*.csv` |
| `tools/build_kelmarsh_notebook.py`, `kaggle_kelmarsh.ipynb` | The GPU run (Kaggle, 2 x T4) that writes `results/kelmarsh_shard*.jsonl` |
| `run_experiment.py` | One rating per row: letter probabilities, exact Shapley values, self-rationale |
| `tools/analyze_kelmarsh.py` | All results (O1-O3) -> `analysis_kelmarsh/` |
| `tools/check_link_recovery.py` | Simulation of slope recovery for the rating-to-state link |
| `tools/make_article_figures.py` | Figures 1-4 of the article |
| `tools/search_literature.py`, `paper/literature_search.csv` | The novelty search (OpenAlex and arXiv, 4 Nov 2026) |
| `data/kelmarsh.json` | 92 evidence months x 6 turbines: eight sentences in three wordings, outcome shares |
| `data/real/` | Monthly state shares per turbine, plant series, report facts (derived from the Kelmarsh data) |
| `results/` | 1,562 LLM ratings with their explanations (the run stopped at its 11-hour limit; 94 ratings of 2024 are missing) |
| `analysis_kelmarsh/` | Every table and number of the article (`summary.md` and CSV files) |
| `tools/zenodo_upload.py` | Uploads this package to Zenodo as a draft (see below) |

## Reproduce

Python 3.11 or later. The analysis needs a CPU only:

```
pip install -r requirements.txt
python tools/analyze_kelmarsh.py results/kelmarsh_shard0.jsonl results/kelmarsh_shard1.jsonl --out analysis_kelmarsh
python tools/check_link_recovery.py
python tools/make_article_figures.py
```

`python tools/analyze_kelmarsh.py --baselines-only ...` gives the results that need no LLM (O1, persistence and the numerical baselines).

From the raw data (about 4 GB):

```
python tools/download_data.py --dest <data folder> --only kelmarsh
python tools/extract_kelmarsh.py --src <data folder>/kelmarsh --out <data folder>/derived
python tools/build_kelmarsh.py --derived <data folder>/derived
python tools/build_kelmarsh_notebook.py
```

Then run `kaggle_kelmarsh.ipynb` on Kaggle with two T4 GPUs ("Save & Run All"); it installs `requirements-gpu.txt`, rates all
1,656 reports (about 54 s per explained rating) and stops after 11 hours, keeping finished rows; attaching its output as input
continues the run. The model and prompt are fixed in the notebook (`microsoft/Phi-3.5-mini-instruct`, float16, scale guide `fleet`).

## Data and licences

- Code: MIT licence (`LICENSE`).
- `data/`, `results/` and `analysis_kelmarsh/` are derived from the Kelmarsh wind farm data by Plumley and Takeuchi (2025),
  https://doi.org/10.5281/zenodo.16807551, licensed CC BY 4.0, and are released under CC BY 4.0 with that attribution.
- The raw Kelmarsh files are not included; `tools/download_data.py` fetches them.

## Citation

See `CITATION.cff`.
