# Kelmarsh results
Ratings: 1562; prompt ['rubric_fleet']


## O1: UGF against observed plant availability
94 months, 186,774 plant-valid intervals.
|   level |   mean_observed |   mean_power |   mean_ugf |    mae |   worst |   mae_power |   worst_power |   corr_power |
|--------:|----------------:|-------------:|-----------:|-------:|--------:|------------:|--------------:|-------------:|
|    0.5  |          0.9999 |       0.9996 |     1      | 0.0001 |  0.0032 |      0.0004 |        0.0049 |        0.176 |
|    0.67 |          0.997  |       0.9966 |     0.9969 | 0.0028 |  0.0665 |      0.0029 |        0.0624 |        0.584 |
|    0.75 |          0.9957 |       0.9876 |     0.9967 | 0.0032 |  0.0565 |      0.0098 |        0.0608 |        0.658 |
|    0.83 |          0.9895 |       0.9446 |     0.9904 | 0.0047 |  0.0649 |      0.0459 |        0.3629 |        0.83  |
|    0.9  |          0.8951 |       0.8745 |     0.9002 | 0.0066 |  0.0711 |      0.0421 |        0.2355 |        0.967 |
|    0.95 |          0.8856 |       0.7747 |     0.8958 | 0.011  |  0.0709 |      0.1237 |        0.3812 |        0.881 |
|    1    |          0.8336 |       0.4891 |     0.8142 | 0.0208 |  0.1061 |      0.3251 |        0.6146 |        0.738 |

## Persistence of each state share
552 turbine-periods.
| state   |   spearman |   within_period |
|:--------|-----------:|----------------:|
| stop    |      0.103 |           0.093 |
| der     |      0.613 |           0.627 |
| full    |      0.417 |           0.289 |

## LLM output checks
Ratings 1562. Letter mass mean 1.000, minimum 1.000, ratings under 0.5: 0. Letters chosen: {'D': 0.528, 'E': 0.211, 'C': 0.209, 'B': 0.053}.
Ratings whose triangle is clipped at 0 or 1: 2.5%. Explained ratings 1184; failures {'kernelshap_exact': 0, 'self_rationale': 0}; rationales naming no valid sentence {'kernelshap_exact': 0, 'self_rationale': 0}.
Wording effect: mean E by wording {0: 0.5271, 1: 0.4395, 2: 0.4631}, largest difference 0.0876, Friedman p = 9.39e-48, 489 turbine-months.

## O2: prediction of the next month
Forward test: 426 turbine-periods from 2019-01; 19 of them with a stopped share above 10% in the next month. Spearman(E, next full share) on all 552: 0.066. Mean E 0.474, mean spread across wordings 0.073.
Slope b: above zero in 71 of 71 fits; mean 3.202, sd 1.724, range 0.408 to 5.333.
Spearman of the rating with this month's shares: stopped -0.503, derated -0.009, full 0.283. AUC, LLM minus persistence, 95% interval -0.115 to +0.179.
| predictor                      |   MAE full share |   CI low |   CI high |   MAE stopped share |   MAE derated share |   Spearman full |   Spearman derated |   AUC long stop |
|:-------------------------------|-----------------:|---------:|----------:|--------------------:|--------------------:|----------------:|-------------------:|----------------:|
| LLM rating + link              |           0.0305 |   0.0228 |    0.0394 |              0.0257 |              0.0109 |          0.0782 |            -0.0904 |          0.7075 |
| Persistence                    |           0.0378 |   0.024  |    0.0537 |              0.0331 |              0.0079 |          0.4025 |             0.5968 |          0.6693 |
| Earlier-periods mean           |           0.0292 |   0.0211 |    0.0388 |              0.0254 |              0.0093 |          0.0912 |            -0.0038 |          0.5256 |
| Ordinal logit on the facts     |           0.0322 |   0.0227 |    0.0436 |              0.0283 |              0.0098 |          0.234  |             0.1891 |          0.5787 |
| Gradient boosting on the facts |           0.0316 |   0.0231 |    0.0408 |              0.0274 |              0.0093 |          0.2981 |             0.3517 |          0.6062 |

| comparison                               |   difference |   CI low |   CI high |   relative |      p |   rank-biserial |   Holm p |
|:-----------------------------------------|-------------:|---------:|----------:|-----------:|-------:|----------------:|---------:|
| LLM minus Persistence                    |      -0.0073 |  -0.0145 |   -0.0011 |    -0.1935 | 0.0365 |          0.1169 |   0.146  |
| LLM minus Earlier-periods mean           |       0.0012 |  -0.0001 |    0.0027 |     0.0419 | 0.443  |          0.0429 |   0.886  |
| LLM minus Ordinal logit on the facts     |      -0.0017 |  -0.0055 |    0.0018 |    -0.0539 | 0.2137 |          0.0695 |   0.6412 |
| LLM minus Gradient boosting on the facts |      -0.0012 |  -0.0038 |    0.0021 |    -0.0367 | 0.5041 |          0.0373 |   0.886  |

## O3: explanations
| explainer        |    n |   faithfulness gain |   irrelevant share U |   stability tau |   seconds |   total hours |
|:-----------------|-----:|--------------------:|---------------------:|----------------:|----------:|--------------:|
| kernelshap_exact | 1184 |              0.0358 |               0.099  |          0.368  |   53.9914 |       17.7572 |
| self_rationale   | 1184 |              0.0274 |               0.0014 |          0.3763 |    0.5724 |        0.1883 |
Shapley minus self-rationale, faithfulness gain: +0.0083, Wilcoxon p = 0.000194, rank-biserial 0.125, n = 1184.

## O3: audit gate (band for next-month full share, turbine level)
|                                 |   score |   width |   hit_rate |   n |
|:--------------------------------|--------:|--------:|-----------:|----:|
| ('kernelshap_exact', 'audited') |  0.1137 |  0.0343 |     0.3286 | 426 |
| ('kernelshap_exact', 'plain')   |  0.1113 |  0.0383 |     0.3709 | 426 |
| ('kernelshap_exact', 'random')  |  0.1114 |  0.033  |     0.3241 | 426 |
| ('self_rationale', 'audited')   |  0.1113 |  0.0382 |     0.3709 | 426 |
| ('self_rationale', 'plain')     |  0.1113 |  0.0383 |     0.3709 | 426 |
| ('self_rationale', 'random')    |  0.1113 |  0.0383 |     0.3709 | 426 |

## O3: audited minus matched random (interval score; turbine intervals by turbine, plant intervals by moving blocks)
| test                      |   difference |   relative |   CI low |   CI high |      p |   rank-biserial |   n |   Holm p |
|:--------------------------|-------------:|-----------:|---------:|----------:|-------:|----------------:|----:|---------:|
| kernelshap_exact, turbine |       0.0023 |     0.0208 |   0.0008 |    0.0038 | 0.0059 |          0.1806 | 426 |   0.0234 |
| kernelshap_exact, plant   |       0.0018 |     0.0047 |  -0.0034 |    0.0072 | 0.1191 |          0.2128 |  71 |   0.3573 |
| self_rationale, turbine   |      -0      |    -0.0004 |  -0.0002 |    0      | 0.2242 |          0.1735 | 426 |   0.4413 |
| self_rationale, plant     |      -0.0002 |    -0.0005 |  -0.0005 |   -0      | 0.2207 |         -0.2148 |  71 |   0.4413 |

## Plant availability at demand 90%
|                                 |   score |   abs_err |   width |   hit_rate |   n |
|:--------------------------------|--------:|----------:|--------:|-----------:|----:|
| ('kernelshap_exact', 'audited') |  0.3886 |    0.1162 |  0.0917 |     0.2254 |  71 |
| ('kernelshap_exact', 'plain')   |  0.3817 |    0.1159 |  0.1023 |     0.2817 |  71 |
| ('kernelshap_exact', 'random')  |  0.3868 |    0.1149 |  0.0883 |     0.2218 |  71 |
| ('self_rationale', 'audited')   |  0.3815 |    0.1158 |  0.1022 |     0.2817 |  71 |
| ('self_rationale', 'plain')     |  0.3817 |    0.1159 |  0.1023 |     0.2817 |  71 |
| ('self_rationale', 'random')    |  0.3817 |    0.1159 |  0.1022 |     0.2817 |  71 |

## O3: sensitivity to the audit percentile (turbine level, audited minus matched random)
| explainer        |   percentile | within wording   |   wordings kept |   audited |   random |   difference |   CI low |   CI high |      p |   rank-biserial |
|:-----------------|-------------:|:-----------------|----------------:|----------:|---------:|-------------:|---------:|----------:|-------:|----------------:|
| kernelshap_exact |          0.5 | False            |          0.5677 |    0.115  |   0.1116 |       0.0034 |   0.0011 |    0.0051 | 0.0004 |          0.2064 |
| kernelshap_exact |          0.6 | False            |          0.6467 |    0.114  |   0.1116 |       0.0024 |   0.0004 |    0.0041 | 0.0109 |          0.1525 |
| kernelshap_exact |          0.7 | False            |          0.6976 |    0.1137 |   0.1113 |       0.0024 |   0.0008 |    0.0039 | 0.0075 |          0.1642 |
| kernelshap_exact |          0.8 | False            |          0.7915 |    0.1133 |   0.1113 |       0.002  |   0.0008 |    0.0032 | 0.0713 |          0.1172 |
| kernelshap_exact |          0.9 | False            |          0.8928 |    0.1118 |   0.1112 |       0.0006 |  -0.0002 |    0.0013 | 0.4357 |          0.0561 |
| self_rationale   |          0.5 | False            |          0.9961 |    0.1113 |   0.1113 |      -0      |  -0.0001 |    0      | 0.2449 |          0.0948 |
| self_rationale   |          0.6 | False            |          0.9961 |    0.1113 |   0.1113 |      -0      |  -0.0001 |    0      | 0.2449 |          0.0948 |
| self_rationale   |          0.7 | False            |          0.9961 |    0.1113 |   0.1113 |      -0      |  -0.0001 |    0      | 0.2449 |          0.0948 |
| self_rationale   |          0.8 | False            |          0.9961 |    0.1113 |   0.1113 |      -0      |  -0.0001 |    0      | 0.2449 |          0.0948 |
| self_rationale   |          0.9 | False            |          0.9961 |    0.1113 |   0.1113 |      -0      |  -0.0001 |    0      | 0.2449 |          0.0948 |

## O3: what the Shapley gate drops
|   wording |   mean_U |   mean_E |   admitted |   n |
|----------:|---------:|---------:|-----------:|----:|
|         0 |    0.135 |    0.529 |      0.415 | 395 |
|         1 |    0.093 |    0.437 |      0.747 | 395 |
|         2 |    0.069 |    0.464 |      0.886 | 394 |
Mean E of admitted wordings 0.4540, of dropped wordings 0.5257.

## O3: exploratory, percentile taken within each wording
| explainer        |   percentile | within wording   |   wordings kept |   audited |   random |   difference |   CI low |   CI high |      p |   rank-biserial |
|:-----------------|-------------:|:-----------------|----------------:|----------:|---------:|-------------:|---------:|----------:|-------:|----------------:|
| kernelshap_exact |          0.6 | True             |          0.7081 |    0.1126 |   0.111  |       0.0016 |   0.0003 |    0.0028 | 0.168  |          0.0853 |
| kernelshap_exact |          0.7 | True             |          0.7093 |    0.1126 |   0.1111 |       0.0015 |   0.0002 |    0.0028 | 0.1464 |          0.0898 |
| kernelshap_exact |          0.8 | True             |          0.8384 |    0.1118 |   0.1113 |       0.0005 |  -0.0002 |    0.0015 | 0.3229 |          0.0669 |
