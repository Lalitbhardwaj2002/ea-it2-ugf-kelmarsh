"""Figures 1-4 of the article, written to paper/figures/. CPU only; none of them needs the LLM results.

    python tools/make_article_figures.py

Fig. 1 methodology flowchart, Fig. 2 operating-state shares, Fig. 3 UGF against observed availability, Fig. 4 persistence.
Each figure is also written as paper/figures_sncs/FigN.tif (600 dpi, LZW) for journal submission (files named Fig1, Fig2, ...; the
series also differ by hatching, line style or marker, so colour is not needed).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "figures")
OUT_SN = os.path.join(ROOT, "paper", "figures_sncs")
BLUE, RED, ORANGE, GREEN, PALE = "#1f6fa8", "#d14a5c", "#f0ad45", "#66a380", "#d4e8f7"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "axes.titlesize": 10.5, "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 500,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


def save(fig, name, n, dpi=None):
    """the article PNG and the journal TIFF (FigN.tif, 600 dpi)"""
    fig.savefig(os.path.join(OUT, name), **({"dpi": dpi} if dpi else {}))
    fig.savefig(os.path.join(OUT_SN, f"Fig{n}.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})


def next_period(p):
    y, m = int(p[:4]), int(p[5:7])
    return f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"


def month_axis(index):
    return np.array([(int(p[:4]) - 2017) * 12 + int(p[5:7]) - 1 for p in index])


def year_ticks(ax):
    ax.set_xticks(np.arange(0, 96, 12), [str(y) for y in range(2017, 2025)])
    ax.set_xlim(-1.5, 96)


def fig_states(sh):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.7, 3.1), gridspec_kw={"width_ratios": [2.1, 1]})
    m = sh.groupby("period")[["p_stop", "p_der"]].mean() * 100
    x = month_axis(m.index)
    a.bar(x, m.p_stop, 0.85, color=RED, label="stopped")
    a.bar(x, m.p_der, 0.85, bottom=m.p_stop, color=ORANGE, hatch="////", edgecolor="white", lw=0, label="derated")
    year_ticks(a)
    a.set_ylabel("Share of plant-valid intervals (%)")
    a.set_title("(a) By month, mean of the six turbines")
    a.legend(loc="upper left")
    w = sh.assign(s=sh.p_stop * sh.n_valid, d=sh.p_der * sh.n_valid).groupby("wt")[["s", "d", "n_valid"]].sum()
    s, d = 100 * w.s / w.n_valid, 100 * w.d / w.n_valid
    x = np.arange(len(w))
    b.bar(x, s, 0.72, color=RED)
    b.bar(x, d, 0.72, bottom=s, color=ORANGE, hatch="////", edgecolor="white", lw=0)
    b.set_xticks(x, [f"KWF{i}" for i in w.index], rotation=45, ha="right", rotation_mode="anchor")
    b.set_title("(b) By turbine, 2017–2024")
    b.margins(x=0.02)
    fig.tight_layout(w_pad=2)
    save(fig, "fig2_states.png", 2)
    plt.close(fig)


def fig_ugf(pc):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.7, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
    p = pc[np.isclose(pc.level, 0.9)].set_index("period")
    on_axis = lambda c: pd.Series(p[c].values, index=month_axis(p.index)).reindex(range(96)).values  # a gap where a month is left out
    style = dict(lw=1.1)
    a.plot(range(96), on_axis("A_ugf"), color=BLUE, label="UGF", **style)
    a.plot(range(96), on_axis("A_obs"), color=ORANGE, ls=":", label="observed, state-coded", **style)
    a.plot(range(96), on_axis("A_power"), color=GREEN, ls="--", label="observed, measured power", **style)
    year_ticks(a)
    a.set_ylabel("Availability")
    a.set_title("(a) Demand 90 % of expected output")
    a.legend(loc="lower left", fontsize=9)
    g = pc.groupby("level")[["A_ugf", "A_obs", "A_power"]].mean()
    x = 100 * g.index.values
    mark = dict(marker="o", ms=3.8, lw=1.3)
    b.plot(x, g.A_ugf, color=BLUE, **mark)
    b.plot(x, g.A_obs, color=ORANGE, ls=":", **{**mark, "marker": "s"})
    b.plot(x, g.A_power, color=GREEN, ls="--", **{**mark, "marker": "^"})
    b.axvspan(92.5, 102, color="#f9eef0", lw=0, zorder=0)
    b.set_xlim(47, 102)
    b.set_xlabel("Demand (% of expected output)")
    b.set_title("(b) Mean over the 94 months")
    fig.tight_layout(w_pad=2)
    save(fig, "fig3_ugf_check.png", 3)
    plt.close(fig)


def fig_persistence(sh):
    n = sh[["period", "wt", "p_stop", "p_der"]].rename(columns={"period": "nx", "p_stop": "s1", "p_der": "d1"})
    t = sh.assign(nx=sh.period.map(next_period)).merge(n, on=["nx", "wt"])
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.3))
    for ax, (x, y, name, tag) in zip(axes, [(t.p_stop, t.s1, "Stopped share", "(a)"), (t.p_der, t.d1, "Derated share", "(b)")]):
        rho = spearmanr(x, y).statistic
        hi = 100 * max(x.max(), y.max()) * 1.06
        ax.plot([0, hi], [0, hi], color="black", lw=0.8, ls="--")
        ax.scatter(100 * x, 100 * y, s=10, color=BLUE, alpha=0.55, lw=0)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{tag} {name}: Spearman {rho:.2f}".replace("-", "−"))
        ax.set_xlabel("This month (%)")
        ax.set_ylabel("Next month (%)")
    fig.tight_layout(w_pad=1)
    save(fig, "fig4_persistence.png", 4)
    plt.close(fig)


# flowchart ------------------------------------------------------------------------------------------------------------
W, H, XC, BW, FS = 155.0, 179.0, 74.0, 100.0, 10.5


def text(ax, x, y, s, **k):
    k.setdefault("ha", "center")
    k.setdefault("va", "center")
    k.setdefault("fontsize", FS)
    k.setdefault("linespacing", 1.28)
    ax.text(x, y, s, **k)


def box(ax, y, h, s, x=XC, w=BW, lw=1.3, ls="-", **k):
    ax.add_patch(Rectangle((x - w / 2, y), w, h, fill=False, lw=lw, ls=ls, ec="black"))
    if s:
        text(ax, x, y + h / 2, s, **k)


def para(ax, y, h, s, x=XC, w=BW, slant=5.0):
    pts = [(x - w / 2 - slant, y + h), (x - w / 2 + slant, y), (x + w / 2 + slant, y), (x + w / 2 - slant, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, lw=1.3, ec="black"))
    text(ax, x, y + h / 2, s)


def diamond(ax, y, h, s, x=XC, w=62.0):
    pts = [(x - w / 2, y + h / 2), (x, y), (x + w / 2, y + h / 2), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=PALE, lw=1.3, ec="black"))
    text(ax, x, y + h / 2, s)


def arrow(ax, *pts):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color="black", lw=1.3, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.5", lw=1.3, color="black", shrinkA=0, shrinkB=0))


def label(ax, y, s):
    text(ax, 11, y, s, fontweight="bold", fontsize=11.5)


def fig_flow():
    fig = plt.figure(figsize=(7.76, 7.76 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    g = 4.0
    y = 1.5
    h = 15.0
    para(ax, y, h, "Input data (10-min resolution, 2017–2024)\nKelmarsh wind farm: six Senvion MM92 turbines, 2.05 MW each\nSCADA signals, expected power, IEC 61400-26-1 status events")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 19.0
    box(ax, y, h, "Pre-processing and operating states\nexternal stops and curtailment removed; all six turbines usable\n"
                  "$r$ = measured / expected power: stopped < 0.1, derated < 0.8, full ≥ 0.8\nstate shares for every turbine and month")
    label(ax, y + h / 2, "C1")
    box(ax, y, h, "Check: UGF,\nEqs. (1)–(3), against\nobserved availability", x=140.5, w=27, lw=1.0, ls=(0, (3, 2)), fontsize=9.5, style="italic")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 11.5
    box(ax, y, h, "Condition report for each turbine and month: three wordings of\nsix measured facts against fleet medians and two static facts")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 15.0
    box(ax, y, h, "LLM rating of each wording\nletter probabilities A–G give expected health $E$, Eq. (4)\nlink from $E$ to state probabilities, Eq. (5), earlier months only")
    label(ax, y + h / 2, "C2")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 11.5
    box(ax, y, h, "Explanation of each rating: exact Shapley values, Eq. (6), and\nthe LLM's own rationale; irrelevant share $U$, Eq. (7)")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 18.0
    diamond(ax, y, h, "$U$ at or below the 70th\npercentile of the month?")
    ym = y + h / 2
    arrow(ax, (XC + 31, ym), (119, ym))
    text(ax, XC + 34.5, ym - 3.0, "No")
    box(ax, ym - 8.5, 17, "Wording not admitted\n(lowest-$U$ wording of\neach turbine is kept)", x=136, w=34, fontsize=10)
    y1 = y + h
    y = y1 + g + 1.0
    arrow(ax, (XC, y1), (XC, y))
    text(ax, XC + 5, y1 + 2.6, "Yes")
    h = 26.5
    box(ax, y, h, "", lw=2.8)
    label(ax, y + h / 2, "C3")
    lines = ["EA-IT2-UGF", "•  triangular health number for each admitted wording, Eq. (8)",
             "•  footprint of uncertainty: envelope and overlap of the triangles",
             "•  $\\alpha$-cut end points through the link and the UGF (Theorem 1)", "•  Karnik–Mendel centroid and band at $\\alpha$ = 0.5"]
    for i, s in enumerate(lines):
        text(ax, XC if i == 0 else XC - BW / 2 + 3, y + 3.8 + 4.7 * i, s, ha="center" if i == 0 else "left", fontweight="bold" if i == 0 else "normal")
    y1 = y + h
    y = y1 + g
    arrow(ax, (XC, y1), (XC, y))
    h = 15.0
    box(ax, y, h, "Evaluation on the next month (forward chaining)\nbaselines: persistence, mean of earlier months, models of the same facts\ncontrols: plain IT2-UGF, matched random-drop; width–miss loss, Eq. (10)")
    box(ax, y, h, "independent\ntruth: next-\nmonth states", x=140.5, w=27, lw=1.0, ls=(0, (3, 2)), fontsize=9.5, style="italic")
    y1 = y + h
    y = y1 + g
    h = 11.5
    xl, xr, wp = XC - 26, XC + 26, 46
    ax.plot([xl, xl], [y1, y], color="black", lw=1.3)
    ax.plot([xr, xr], [y1, y], color="black", lw=1.3)
    para(ax, y, h, "Turbine level: band for\nnext full-output share", x=xl, w=wp, slant=3)
    para(ax, y, h, "Plant level: band for\navailability, 90% demand", x=xr, w=wp, slant=3)
    print("flowchart bottom", y + h, "of", H)
    save(fig, "fig1_methodology.png", 1, dpi=400)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_SN, exist_ok=True)
    sh = pd.read_csv(os.path.join(ROOT, "data/real/kelmarsh_shares.csv"))
    pc = pd.read_csv(os.path.join(ROOT, "data/real/kelmarsh_plant.csv"))
    fig_states(sh)
    fig_flow()
    fig_ugf(pc)
    fig_persistence(sh)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
