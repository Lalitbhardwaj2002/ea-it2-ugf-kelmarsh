"""Can the rating-to-state link be identified when stopped and derated shares are this small? CPU only; simulated ratings, so no LLM result.

    python tools/check_link_recovery.py

Ratings are drawn with a known slope b, shares are drawn from the link at the imbalance seen at Kelmarsh, and the slope is fitted again.
Noise = 0 gives clean shares. Noise = 1.5 adds a turbine-month shock on the logit scale, which mimics long outages that no rating could foresee.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from analyze_kelmarsh import fit_link, link_probs

P_STOP, P_FULL = 0.0185, 0.967  # observed at Kelmarsh
INTERVALS, E_MEAN, E_SD, SIMS = 1990, 0.55, 0.10, 200
logit = lambda p: np.log(p / (1 - p))


def main():
    rows = []
    for b in (1.0, 3.0, 6.0):
        a1, a2 = logit(1 - P_STOP) - b * E_MEAN, logit(P_FULL) - b * E_MEAN
        for n in (126, 552):
            for noise in (0.0, 1.5):
                rng = np.random.default_rng(int(b * 10) + n)
                fits = []
                for _ in range(SIMS):
                    e = np.clip(rng.normal(E_MEAN, E_SD, n), 0, 1)
                    q = link_probs((a1, a2, b), e + rng.normal(0, noise, n) / b)
                    shares = np.array([rng.multinomial(INTERVALS, x / x.sum()) for x in q]) / INTERVALS
                    fits.append(fit_link(e, shares)[2])
                f = np.array(fits)
                rows.append({"true b": b, "turbine-months": n, "noise": noise, "mean b": f.mean(), "sd b": f.std(), "share b = 0": float((f < 1e-6).mean()), "share within 50 %": float((np.abs(f - b) <= 0.5 * b).mean())})
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows).round(3)
    os.makedirs(os.path.join(ROOT, "analysis_kelmarsh"), exist_ok=True)
    t.to_csv(os.path.join(ROOT, "analysis_kelmarsh", "o2_link_recovery.csv"), index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
