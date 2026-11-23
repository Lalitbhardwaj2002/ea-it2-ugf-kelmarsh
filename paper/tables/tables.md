# Result tables of the article

Made by `tools/make_tables.py` from `analysis_kelmarsh/`. Each table is also a CSV file in this folder.

**Table 4 UGF against observed plant availability: means over 94 months, and mean absolute errors with the worst month in brackets**

| Demand (%) | UGF | State-coded | Measured | Error, state-coded | Error, measured |
|---|---|---|---|---|---|
| 50 | 1.0000 | 0.9999 | 0.9996 | 0.0001 (0.0032) | 0.0004 (0.0049) |
| 67 | 0.9969 | 0.9970 | 0.9966 | 0.0028 (0.0665) | 0.0029 (0.0624) |
| 75 | 0.9967 | 0.9957 | 0.9876 | 0.0032 (0.0565) | 0.0098 (0.0608) |
| 83 | 0.9904 | 0.9895 | 0.9446 | 0.0047 (0.0649) | 0.0459 (0.3629) |
| 90 | 0.9002 | 0.8951 | 0.8745 | 0.0066 (0.0711) | 0.0421 (0.2355) |
| 95 | 0.8958 | 0.8856 | 0.7747 | 0.0110 (0.0709) | 0.1237 (0.3812) |
| 100 | 0.8142 | 0.8336 | 0.4891 | 0.0208 (0.1061) | 0.3251 (0.6146) |

**Table 5 Prediction of the next month on the forward test (426 turbine-months). Errors are mean absolute errors of the state shares, with the 95% interval for the full-output share. ρ: Spearman correlation. AUC: for a stopped share above 10% in the next month (19 cases)**

| Predictor | Full output | Stopped | Derated | ρ, full | ρ, derated | AUC |
|---|---|---|---|---|---|---|
| LLM rating + link | 0.0305 (0.0228–0.0394) | 0.0257 | 0.0109 | 0.08 | −0.09 | 0.71 |
| Mean of earlier months | 0.0292 (0.0211–0.0388) | 0.0254 | 0.0093 | 0.09 | 0.00 | 0.53 |
| Persistence | 0.0378 (0.0240–0.0537) | 0.0331 | 0.0079 | 0.40 | 0.60 | 0.67 |
| Ordinal logit | 0.0322 (0.0227–0.0436) | 0.0283 | 0.0098 | 0.23 | 0.19 | 0.58 |
| Gradient boosting | 0.0316 (0.0231–0.0408) | 0.0274 | 0.0093 | 0.30 | 0.35 | 0.61 |

**Table 6 Explanations of the 1,184 ratings in the test months. Faithfulness is the deletion-sensitivity gain over random removal. Higher faithfulness and a lower irrelevant share are better**

| Explainer | Faithfulness | Irrelevant share | Stability (τ) | Seconds |
|---|---|---|---|---|
| Exact Shapley values | 0.036 | 0.099 | 0.37 | 54.0 |
| LLM self-rationale | 0.027 | 0.001 | 0.38 | 0.6 |

**Table 7 Bands on the forward test. Turbine level: next-month full-output share (426 turbine-months). Plant level: availability at 90% demand (71 months). Loss: Eq. (10), lower is better. Error: absolute error of the point estimate. Hit: share of observed values inside the band, with no nominal level. Rows 2 and 3 are the primary comparison, the Shapley gate against random removal of the same count**

| Band | Turbine loss | Turbine width | Turbine hit | Plant loss | Plant error | Plant width | Plant hit |
|---|---|---|---|---|---|---|---|
| Plain IT2-UGF | 0.1113 | 0.038 | 0.37 | 0.382 | 0.116 | 0.102 | 0.28 |
| EA-IT2-UGF, Shapley gate | 0.1137 | 0.034 | 0.33 | 0.389 | 0.116 | 0.092 | 0.23 |
| Random-drop (Shapley count) | 0.1114 | 0.033 | 0.32 | 0.387 | 0.115 | 0.088 | 0.22 |
| EA-IT2-UGF, self-rationale gate | 0.1113 | 0.038 | 0.37 | 0.382 | 0.116 | 0.102 | 0.28 |
| Random-drop (self-rationale count) | 0.1113 | 0.038 | 0.37 | 0.382 | 0.116 | 0.102 | 0.28 |
