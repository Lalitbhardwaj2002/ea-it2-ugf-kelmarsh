"""Step 2 of the Kelmarsh case study: operating states, plant check, evidence dossiers and the benchmark file.

    python tools/build_kelmarsh.py [--derived E:\\xfmss_data\\derived] [--full 0.8]

Reads the output of tools/extract_kelmarsh.py and writes (raw data stays on E:, these small files stay in the project):
    data/real/kelmarsh_shares.csv     share of time stopped / derated / full per turbine and month
    data/real/kelmarsh_features.csv   monthly condition indicators per turbine
    data/real/kelmarsh_plant.csv      observed plant availability against the UGF, per month and demand
    data/kelmarsh.json                one "plant" per window: evidence from one month, outcome from the next

Operating state of a turbine in a 10-minute interval, from r = power / expected power (learned power curve):
    stopped r < 0.1;  derated 0.1 <= r < full;  full r >= full (default 0.8).
Valid = enough wind to expect at least 10 % of rated power, wind below 20 m/s, and no external cause: intervals inside
a stop with IEC category Out of Electrical Specification, Requested Shutdown or Out of Environmental Specification, or
with curtailment losses, are left out. An interval counts only if all six turbines are valid at the same time, so that
turbine shares and observed plant availability describe exactly the same intervals. A month needs at least MIN_INTERVALS
plant-valid intervals to count (August and September 2017 do not reach it).
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from xfmss import benchmark as bm
from xfmss import ugf

RATED = 2050.0
START = "2017-01-01"  # turbine 6 has almost no usable data in 2016
EXTERNAL = {"Out of Electrical Specification", "Requested Shutdown", "Out of Environmental Specification"}
STATIC = {1: (78.5, 145.6, "SEN 93420"), 2: (78.5, 156.6, "SEN 93421"), 3: (68.5, 153.5, "SEN 93422"),
          4: (78.5, 146.3, "SEN 93423"), 5: (78.5, 142.9, "SEN 93424"), 6: (68.5, 135.0, "SEN 93425")}  # Kelmarsh_WT_static.csv
LEVELS = [0.5, 0.67, 0.75, 0.83, 0.9, 0.95, 1.0]
MIN_INTERVALS = 500
MIN_HIGH_LOAD = 6  # one hour at high load, below which a monthly temperature is not reported
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def period_of(t):
    return t.dt.to_period("M").astype(str)


def period_words(p):
    return f"{MONTHS[int(p[5:7]) - 1]} {p[:4]}"


def next_period(p):
    y, m = int(p[:4]), int(p[5:7])
    return f"{y + 1}-01" if m == 12 else f"{y}-{m + 1:02d}"


def states(d, ev, full):
    """Adds r, valid and state (0 stopped, 1 derated, 2 full) to the 10-minute table."""
    d = d.sort_values(["wt", "time"]).reset_index(drop=True)
    d["external"] = False
    stops = ev[(ev["Status"] == "Stop") & ev["IEC category"].isin(EXTERNAL) & ev.start.notna() & ev.end.notna()]
    for wt, g in d.groupby("wt"):
        t = g.time.values
        base = g.index[0]
        mask = np.zeros(len(t), bool)
        for s, e in zip(stops[stops.wt == wt].start.values, stops[stops.wt == wt].end.values):
            i, j = np.searchsorted(t, s - np.timedelta64(10, "m")), np.searchsorted(t, e, side="right")
            mask[i:j] = True
        d.loc[base:base + len(t) - 1, "external"] = mask
    d["r"] = d.power / d.pot_learned
    d["valid"] = (d.pot_learned >= 0.1 * RATED) & (d.wind < 20) & ~d.external & ~(d.lost_curt > 0) & d.power.notna()
    d["state"] = np.where(d.r < 0.1, 0, np.where(d.r < full, 1, 2))
    n = d[d.valid].groupby("time").wt.nunique()
    d["plant_valid"] = d.time.isin(n[n == 6].index)
    return d


def shares(v):
    g = v.groupby(["period", "wt"])
    out = g.state.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    out = out.rename(columns={0: "p_stop", 1: "p_der", 2: "p_full"}).reset_index()
    out["n_valid"] = g.size().values
    return out


def _plant(period_shares, delta):
    comps = [{"id": f"KWF{r.wt}", "type": "wind turbine", "cap": RATED / 1000, "unit": "MW", "degraded_fraction": round(float(delta), 3)}
             for r in period_shares.itertuples()]
    return {"subsystems": [{"name": "Turbines", "components": comps}], "demand": 0.9 * 6 * RATED / 1000,
            "structure": {"type": "parallel", "children": [{"asset": c["id"]} for c in comps]}}


def plant_check(v, sh, delta):
    """Observed availability from the simultaneous states against the UGF built from the same intervals' turbine shares."""
    rows = []
    v = v.assign(out=np.select([v.state == 0, v.state == 1], [0.0, delta], 1.0))
    for p, g in v.groupby("period"):
        w = g.pivot(index="time", columns="wt", values="out")
        t = g.groupby("time")
        ratio = t.power.sum().clip(lower=0) / t.pot_learned.sum()  # measured plant power over expected plant power
        s = sh[sh.period == p]
        plant = _plant(s, delta)
        probs = {f"KWF{r.wt}": [r.p_stop, r.p_der, r.p_full] for r in s.itertuples()}
        for lv in LEVELS:
            rows.append({"period": p, "level": lv, "A_obs": float((w.sum(axis=1) >= 6 * lv - 1e-9).mean()), "A_power": float((ratio >= lv - 1e-9).mean()),
                         "A_ugf": ugf.availability_from_probs(plant, probs, demand=6 * lv * RATED / 1000), "n": len(w)})
    return pd.DataFrame(rows)


def features(d, v, ev):
    """Monthly condition indicators; temperatures are compared at high load so they do not simply follow the wind."""
    forced = ev[(ev["Status"] == "Stop") & (ev["IEC category"] == "Forced outage") & ev.start.notna()].copy()
    forced["period"] = period_of(forced.start)
    rows = []
    for (p, wt), g in d.groupby(["period", "wt"]):
        hi = g[g.power >= 0.7 * RATED]
        if len(hi) < MIN_HIGH_LOAD:
            hi = hi.iloc[:0]  # too little high-load running for a temperature; the mean of nothing is NaN
        u = v[(v.period == p) & (v.wt == wt)]
        rows.append({"period": p, "wt": wt, "forced_stops": int(((forced.period == p) & (forced.wt == wt)).sum()),
                     "stopped_h": float((u.state == 0).sum() / 6), "derated_pct": 100 * float((u.state == 1).mean()),
                     "gear_oil_t": float(hi.gear_oil_t.mean()), "gen_bear_t": float(hi[["gen_bear_front_t", "gen_bear_rear_t"]].mean(axis=1).mean()),
                     "transf_t": float(hi.transf_t.mean())})
    return pd.DataFrame(rows)


TEMPLATES = {
    "forced_stops": ["Forced outages stopped the turbine {x} times in {y}; the median for the six turbines was {m}.",
                     "The event log shows {x} forced-outage stops during {y}, against a fleet median of {m}.",
                     "It had {x} forced stops during {y}, while the fleet median was {m} stops."],
    "stopped_h": ["It stood stopped for {x} hours of usable wind in {y}; the fleet median was {m} hours.",
                  "Stopped time in usable wind reached {x} hours in {y}, against a fleet median of {m} hours.",
                  "It lost {x} hours of usable wind while stopped in {y}; the fleet median was {m} hours."],
    "derated_pct": ["It ran below 80% of its expected power for {x}% of the usable wind in {y}; the fleet median was {m}%.",
                    "Output stayed under 80% of the expected power for {x}% of usable wind in {y}, against a fleet median of {m}%.",
                    "For {x}% of usable wind in {y} it delivered under 80% of expected power, compared with a fleet median of {m}%."],
    "gear_oil_t": ["Gear oil temperature at high load averaged {x} °C in {y}; the fleet median was {m} °C.",
                   "At high load the gear oil ran at {x} °C during {y}, against a fleet median of {m} °C.",
                   "High-load gear oil temperature was {x} °C in {y}, compared with a fleet median of {m} °C."],
    "gen_bear_t": ["Generator bearing temperature at high load averaged {x} °C in {y}; the fleet median was {m} °C.",
                   "At high load the generator bearings ran at {x} °C during {y}, against a fleet median of {m} °C.",
                   "High-load generator bearing temperature was {x} °C in {y}, compared with a fleet median of {m} °C."],
    "transf_t": ["Transformer temperature at high load averaged {x} °C in {y}; the fleet median was {m} °C.",
                 "At high load the transformer ran at {x} °C during {y}, against a fleet median of {m} °C.",
                 "High-load transformer temperature was {x} °C in {y}, compared with a fleet median of {m} °C."],
}
ROUND = {"forced_stops": 0, "stopped_h": 0, "derated_pct": 1, "gear_oil_t": 1, "gen_bear_t": 1, "transf_t": 1}
NO_READING = {"gear_oil_t": "gear oil", "gen_bear_t": "generator bearing", "transf_t": "transformer"}


def cues(fp, wt, words):
    r = fp[fp.wt == wt].iloc[0]
    out = []
    for k, tpl in TEMPLATES.items():
        if pd.isna(r[k]):  # still a true statement, and the report keeps its eight sentences
            n, med = NO_READING[k], fp[k].median()
            tail = f"; the fleet median was {round(float(med), 1)} °C." if pd.notna(med) else "."
            out.append({"cue_id": k, "kind": "causal", "value": None, "templates": [
                f"No high-load {n} temperature is available for {words}{tail}",
                f"There is no {n} temperature at high load for {words}{tail}",
                f"The records for {words} hold no high-load {n} temperature{tail}"]})
            continue
        x, m = round(float(r[k]), ROUND[k]), round(float(fp[k].median()), ROUND[k])
        x, m = (int(x), int(m)) if ROUND[k] == 0 else (x, m)
        out.append({"cue_id": k, "kind": "causal", "value": x, "templates": [t.format(x=x, m=m, y=words) for t in tpl]})
    hub, elev, serial = STATIC[wt]
    out.append({"cue_id": "hub_height", "kind": "distractor", "templates": [
        f"Its hub stands {hub} m above the ground, carrying a rotor 92 m in diameter.",
        f"The rotor is 92 m across and the hub height is {hub} m above ground level.",
        f"The tower puts the hub {hub} m above ground, under a 92 m diameter rotor."]})
    out.append({"cue_id": "site_id", "kind": "distractor", "templates": [
        f"It is registered as {serial} and stands at an elevation of {elev} m on the site.",
        f"The site register lists it as {serial}, at an elevation of {elev} m above sea level.",
        f"Its serial number is {serial}, and the foundation sits at {elev} m elevation."]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default=r"E:\xfmss_data\derived")
    ap.add_argument("--full", type=float, default=0.8)
    a = ap.parse_args()
    d = pd.read_pickle(os.path.join(a.derived, "kelmarsh_10min.pkl"))
    ev = pd.read_pickle(os.path.join(a.derived, "kelmarsh_status.pkl"))
    d = d[d.time >= START].copy()
    d["period"] = period_of(d.time)
    d = states(d, ev, a.full)
    v = d[d.valid & d.plant_valid]
    sh = shares(v)
    print("months left out:", sh[sh.n_valid < MIN_INTERVALS].period.unique().tolist())
    sh = sh[sh.n_valid >= MIN_INTERVALS].reset_index(drop=True)
    delta = float(v[v.state == 1].r.mean())
    print(f"10-minute records {d.time.nunique():,}; plant-valid {v.time.nunique():,} ({v.time.nunique() / d.time.nunique():.1%}); "
          f"stopped {(v.state == 0).mean():.4f}, derated {(v.state == 1).mean():.4f}, full {(v.state == 2).mean():.4f}; delta {delta:.3f}")
    pc = plant_check(v[v.period.isin(set(sh.period))], sh, delta)
    os.makedirs("data/real", exist_ok=True)
    tag = "" if a.full == 0.8 else f"_full{int(a.full * 100)}"
    sh.to_csv(f"data/real/kelmarsh_shares{tag}.csv", index=False)
    pc.to_csv(f"data/real/kelmarsh_plant{tag}.csv", index=False)
    print(pc.groupby("level").apply(lambda g: pd.Series({"mean_obs": g.A_obs.mean(), "mean_ugf": g.A_ugf.mean(), "mae": (g.A_obs - g.A_ugf).abs().mean(), "max_err": (g.A_obs - g.A_ugf).abs().max()}), include_groups=False).round(4).to_string())
    if tag:
        return
    fe = features(d, v, ev)
    fe.to_csv("data/real/kelmarsh_features.csv", index=False)
    print("no high-load temperature:", fe[fe.period.isin(set(sh.period))][list(NO_READING)].isna().sum().to_dict())
    bench = []
    for p in sorted(sh.period.unique()):
        q = next_period(p)
        nxt = sh[sh.period == q]
        if len(nxt) < 6:
            continue
        plant = _plant(nxt, delta)
        plant.update({"id": f"kelmarsh_{p}", "name": f"Kelmarsh, evidence {p}, outcome {q}", "unit": "MW", "installed_mw": 12.3,
                      "data_source": "Plumley and Takeuchi (2025), doi:10.5281/zenodo.16807551"})
        for c in plant["subsystems"][0]["components"]:
            wt = int(c["id"][-1])
            r = nxt[nxt.wt == wt].iloc[0]
            c["state_probs_true"] = [round(float(r.p_stop), 5), round(float(r.p_der), 5), round(float(r.p_full), 5)]
            c["h_true"] = round(ugf.health_from_probs(c["state_probs_true"]), 3)
            c["cues"] = cues(fe[fe.period == p], wt, period_words(p))
        bench.append(plant)
    bm.save(bench, "data/kelmarsh.json")
    print(f"benchmark: {len(bench)} windows, {sum(len(b['subsystems'][0]['components']) for b in bench)} turbine-periods")


if __name__ == "__main__":
    main()
