# EA-IT2-UGF: explanation-audited interval type-2 fuzzy UGF on the Kelmarsh wind farm

Code, derived data and outputs for the article

> Bhardwaj L, Sharma MK. *Does Explanation-Based Screening Improve LLM-Based Fuzzy Forecasts of Wind-Farm Availability? A Controlled Test* (manuscript, 2026).

The study asks whether screening large language model (LLM) health ratings of wind turbines by their explanations improves
next-month availability forecasts beyond unscreened ratings and beyond random removal of the same number of ratings. An LLM
(Phi-3.5-mini) rates each turbine from monthly condition reports in three wordings, exact Shapley values measure how much of each
rating's attribution falls on irrelevant sentences, an audit gate removes the worst wordings, and the retained ratings form interval
type-2 (IT2) fuzzy health numbers that a universal generating function (UGF) propagates exactly into availability bands. On eight
years of Kelmarsh data the gate narrowed the bands by 10 % but did not beat matched random removal.

## Contents

| Path | What it is |
|---|---|
| `xfmss/` | Python package: UGF (`ugf.py`), IT2 numbers and propagation (`it2.py`), LLM reading of letter probabilities (`llm.py`), explainers (`explain.py`), metrics (`metrics.py`), report assembly (`benchmark.py`). `estimators.py`, `windfarm.py` and `dataio.py` serve a synthetic benchmark of a later study and are not used here; the estimators of this study are in `tools/analyze_kelmarsh.py` |
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
| `data/real/` | Monthly state shares per turbine, plant series, report facts, derived from the Kelmarsh data (`*_full90.csv`: full state at 90 % instead of 80 %, a sensitivity case; `zenodo_records/`: metadata of the source record) |
| `results/` | 1,562 LLM ratings; the 1,184 from January 2019 on carry exact Shapley values and self-rationales (the first two years only train the link). The run stopped at its 11-hour limit; 94 ratings of 2024 are missing |
| `analysis_kelmarsh/` | The result tables and numbers of the article (`summary.md` and CSV files); data-preparation counts are printed by `tools/build_kelmarsh.py` |
| `tools/zenodo_upload.py` | Alternative route: uploads this package to Zenodo as a draft with a reserved DOI; `--publish <id>` publishes it (instructions at the top of the file). The archived version was made from the GitHub release; use one route per version, or Zenodo makes two records |

## Reproduce

Python 3.11 or later; run every command from the package folder. The archived outputs were reproduced with Python 3.14.6,
numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2 and tabulate 0.10.0; other versions may differ in
the last digits. The figures use the Arial font; without it matplotlib falls back to another font. The analysis needs a CPU only:

```
pip install -r requirements.txt
python tools/analyze_kelmarsh.py results/kelmarsh_shard0.jsonl results/kelmarsh_shard1.jsonl --out analysis_kelmarsh
python tools/check_link_recovery.py
python tools/make_article_figures.py
```

`python tools/analyze_kelmarsh.py --baselines-only --out analysis_baselines` gives the results that need no LLM (O1, persistence
and the numerical baselines). Use a separate `--out` folder: the default `analysis_kelmarsh` would be overwritten with a shorter summary.

From the raw data (about 4 GB). The scripts' default data folder is `E:\xfmss_data` (`$XFMSS_DATA` for `download_data.py`), so
pass the paths as shown:

```
python tools/download_data.py --dest <data folder> --only kelmarsh
python tools/extract_kelmarsh.py --src <data folder>/kelmarsh --out <data folder>/derived
python tools/build_kelmarsh.py --derived <data folder>/derived
python tools/build_kelmarsh.py --derived <data folder>/derived --full 0.9
python tools/build_kelmarsh_notebook.py
```

The second `build_kelmarsh.py` call writes the sensitivity files `data/real/*_full90.csv`. Then run `kaggle_kelmarsh.ipynb` on
Kaggle with two T4 GPUs ("Save & Run All"). It installs the packages listed in `requirements-gpu.txt`, rates the 552 turbine-month
reports in three wordings (1,656 ratings; about 54 s per explained rating) and stops after 11 hours, keeping finished rows; attach
its output as input and run it again to finish. Unzip its `kelmarsh_results.zip` in the package folder (it holds `results/` and
`logs/`). The model and prompt are fixed in the notebook (`microsoft/Phi-3.5-mini-instruct`, float16, scale guide `fleet`). The
archived run (Kaggle, November 2026) used inseq 0.7.1 and the then-current transformers and shap; exact versions were not recorded.

`run_experiment.py` and `xfmss/` are exactly the code embedded in the notebook that produced `results/`, so they are archived
unchanged. Their comments also mention a synthetic benchmark (`data/benchmark.json`), a pre-flight script (`check_methods.py`) and
Inseq explainers that are not part of this package. Always pass `--bench data/kelmarsh.json`: if the `--bench` file does not exist,
`run_experiment.py` creates and rates a synthetic benchmark instead.

## Data and licences

- Code: MIT licence (`LICENSE`).
- `data/`, `results/` and `analysis_kelmarsh/` are derived from the Kelmarsh wind farm data by Plumley and Takeuchi (2025),
  https://doi.org/10.5281/zenodo.16807551, licensed CC BY 4.0, and are released under CC BY 4.0 with that attribution.
- `paper/literature_search.csv` lists bibliographic records from OpenAlex and arXiv (metadata, CC0).
- The raw Kelmarsh files are not included; `tools/download_data.py` fetches them.

## Citation

See `CITATION.cff`.
