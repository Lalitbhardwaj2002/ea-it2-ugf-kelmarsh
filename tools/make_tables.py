"""Tables 4-7 of the article (the result tables) from analysis_kelmarsh/, as CSV files and one Markdown page, in paper/tables/.

    python tools/make_tables.py

Tables 1-3 of the article describe the literature, the data and the compared methods; they are text and not computed. The numbers and
their formatting here are the ones printed in the article.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = os.path.join(ROOT, "analysis_kelmarsh")
OUT = os.path.join(ROOT, "paper", "tables")
f4 = lambda v: f"{v:.4f}"
rho = lambda v: "0.00" if abs(v) < 0.005 else f"{v:.2f}".replace("-", "−")


def tables():
    t = {}
    o1 = pd.read_csv(os.path.join(A, "o1_ugf_check.csv"))
    t["Table4_ugf_check"] = (
        "Table 4 UGF against observed plant availability: means over 94 months, and mean absolute errors with the worst month in brackets",
        ["Demand (%)", "UGF", "State-coded", "Measured", "Error, state-coded", "Error, measured"],
        [[f"{100 * r.level:.0f}", f4(r.mean_ugf), f4(r.mean_observed), f4(r.mean_power), f"{r.mae:.4f} ({r.worst:.4f})",
          f"{r.mae_power:.4f} ({r.worst_power:.4f})"] for r in o1.itertuples()])
    o2 = pd.read_csv(os.path.join(A, "o2_prediction.csv")).set_index("predictor")
    row = lambda name, label: [label, f"{o2.loc[name, 'MAE full share']:.4f} ({o2.loc[name, 'CI low']:.4f}–{o2.loc[name, 'CI high']:.4f})",
                               f4(o2.loc[name, "MAE stopped share"]), f4(o2.loc[name, "MAE derated share"]), rho(o2.loc[name, "Spearman full"]),
                               rho(o2.loc[name, "Spearman derated"]), f"{o2.loc[name, 'AUC long stop']:.2f}"]
    t["Table5_next_month"] = (
        "Table 5 Prediction of the next month on the forward test (426 turbine-months). Errors are mean absolute errors of the state shares, with the 95% "
        "interval for the full-output share. ρ: Spearman correlation. AUC: for a stopped share above 10% in the next month (19 cases)",
        ["Predictor", "Full output", "Stopped", "Derated", "ρ, full", "ρ, derated", "AUC"],
        [row("LLM rating + link", "LLM rating + link"), row("Earlier-periods mean", "Mean of earlier months"), row("Persistence", "Persistence"),
         row("Ordinal logit on the facts", "Ordinal logit"), row("Gradient boosting on the facts", "Gradient boosting")])
    ex = pd.read_csv(os.path.join(A, "o3_explainers.csv")).set_index("explainer")
    er = lambda m, label: [label, f"{ex.loc[m, 'faithfulness gain']:.3f}", f"{ex.loc[m, 'irrelevant share U']:.3f}", f"{ex.loc[m, 'stability tau']:.2f}",
                           f"{ex.loc[m, 'seconds']:.1f}"]
    t["Table6_explanations"] = (
        "Table 6 Explanations of the 1,184 ratings in the test months. Faithfulness is the deletion-sensitivity gain over random removal. Higher "
        "faithfulness and a lower irrelevant share are better",
        ["Explainer", "Faithfulness", "Irrelevant share", "Stability (τ)", "Seconds"],
        [er("kernelshap_exact", "Exact Shapley values"), er("self_rationale", "LLM self-rationale")])
    tu = pd.read_csv(os.path.join(A, "o3_audit_turbines.csv")).groupby(["explainer", "est"])[["score", "width", "covered"]].mean()
    pl = pd.read_csv(os.path.join(A, "o3_audit_plant.csv")).groupby(["explainer", "est"])[["score", "abs_err", "width", "covered"]].mean()
    au = lambda m, e, label: [label, f"{tu.loc[(m, e), 'score']:.4f}", f"{tu.loc[(m, e), 'width']:.3f}", f"{tu.loc[(m, e), 'covered']:.2f}",
                              f"{pl.loc[(m, e), 'score']:.3f}", f"{pl.loc[(m, e), 'abs_err']:.3f}", f"{pl.loc[(m, e), 'width']:.3f}", f"{pl.loc[(m, e), 'covered']:.2f}"]
    s, r = "kernelshap_exact", "self_rationale"
    t["Table7_bands"] = (
        "Table 7 Bands on the forward test. Turbine level: next-month full-output share (426 turbine-months). Plant level: availability at 90% demand "
        "(71 months). Loss: Eq. (10), lower is better. Error: absolute error of the point estimate. Hit: share of observed values inside the band, with "
        "no nominal level. Rows 2 and 3 are the primary comparison, the Shapley gate against random removal of the same count",
        ["Band", "Turbine loss", "Turbine width", "Turbine hit", "Plant loss", "Plant error", "Plant width", "Plant hit"],
        [au(s, "plain", "Plain IT2-UGF"), au(s, "audited", "EA-IT2-UGF, Shapley gate"), au(s, "random", "Random-drop (Shapley count)"),
         au(r, "audited", "EA-IT2-UGF, self-rationale gate"), au(r, "random", "Random-drop (self-rationale count)")])
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    md = ["# Result tables of the article", "", "Made by `tools/make_tables.py` from `analysis_kelmarsh/`. Each table is also a CSV file in this folder.", ""]
    for name, (caption, header, rows) in tables().items():
        pd.DataFrame(rows, columns=header).to_csv(os.path.join(OUT, f"{name}.csv"), index=False, encoding="utf-8")
        md += [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        md += ["| " + " | ".join(r) + " |" for r in rows] + [""]
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
