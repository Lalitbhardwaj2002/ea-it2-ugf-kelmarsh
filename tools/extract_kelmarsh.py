"""Step 1 of the Kelmarsh case study: read the raw Zenodo zips once and keep what the study needs.

    python tools/extract_kelmarsh.py [--src E:\\xfmss_data\\kelmarsh] [--out E:\\xfmss_data\\derived]

For each turbine and year it reads the 10-minute SCADA table and the status (event) table directly from the zip,
without unpacking, saves each year, then writes kelmarsh_10min.pkl (selected signals) and kelmarsh_status.pkl (events).
"""
import argparse, glob, io, os, re, zipfile
import pandas as pd

# output name -> start of the column name in the Greenbyte export
SIGNALS = {"wind": "Wind speed (m/s)", "power": "Power (kW)", "pot_learned": "Potential power learned PC (kW)",
           "pot_default": "Potential power default PC (kW)", "setpoint": "Turbine Power setpoint (kW)",
           "lost_curt": "Lost Production to Curtailment (Total) (kWh)", "lost_down": "Lost Production to Downtime (kWh)",
           "lost_perf": "Lost Production to Performance (kWh)",
           "gear_oil_t": "Gear oil temperature", "gear_inlet_t": "Gear oil inlet temperature",
           "gear_inlet_p": "Gear oil inlet pressure", "gen_bear_front_t": "Generator bearing front temperature",
           "gen_bear_rear_t": "Generator bearing rear temperature", "main_front_t": "Front bearing temperature",
           "main_rear_t": "Rear bearing temperature", "rotor_bear_t": "Rotor bearing temp",
           "stator_t": "Stator temperature 1", "transf_t": "Transformer temperature",
           "ambient_t": "Nacelle ambient temperature", "drive_acc": "Drive train acceleration",
           "tower_acc_x": "Tower Acceleration X", "metal_count": "Metal particle count",
           "metal_counter": "Metal particle count counter"}


def header(z, name):
    """Greenbyte files start with '#' lines. Returns how many, and the last one (SCADA tables keep their column names there)."""
    with z.open(name) as f:
        skip, last = 0, None
        for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
            if not line.startswith("#"):
                break
            skip, last = skip + 1, line
    return skip, last


def columns(line):
    return list(pd.read_csv(io.StringIO(line[1:].strip()), nrows=0).columns)


def wanted(cols):
    """Map output names to the exact columns of this file (units and spellings changed between years)."""
    out = {}
    for key, start in SIGNALS.items():
        if key == "metal_count":  # 'Metal particle count' has no unit; do not match the '... counter' column
            hit = [c for c in cols if c == start]
        else:
            hit = [c for c in cols if c.startswith(start) and "," not in c]
        if hit:
            out[hit[0]] = key
    return out


def read_turbine(z, name, skip, last, wt):
    """One SCADA table -> one row per timestamp.

    The 2023 and 2024 exports repeat every timestamp many times, out of order, and many of the repeats are mostly
    empty, so rows are merged per timestamp (first non-missing value of each signal). Repeats that disagree are counted.
    """
    cols = columns(last)
    mp = wanted(cols)
    parts = []
    with z.open(name) as f:
        for ch in pd.read_csv(f, skiprows=skip, header=None, names=cols, usecols=["Date and time"] + list(mp),
                              na_values=["NaN"], chunksize=200_000, encoding="utf-8", encoding_errors="replace"):
            ch = ch.rename(columns={**mp, "Date and time": "time"})
            sig = [c for c in ch.columns if c != "time"]
            for c in sig:
                ch[c] = pd.to_numeric(ch[c], errors="coerce").astype("float32")
            parts.append(ch.dropna(subset=sig, how="all"))
    d = pd.concat(parts, ignore_index=True)
    raw = len(d)
    d["time"] = pd.to_datetime(d["time"])
    conflict = float((d.groupby("time")["power"].nunique() > 1).mean())
    d = d.groupby("time", sort=True).first().reset_index()
    d["wt"] = wt
    print(f"  WT{wt}: {raw:,} non-empty rows -> {len(d):,} timestamps; conflicting power at {conflict:.3%} of them", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"E:\xfmss_data\kelmarsh")
    ap.add_argument("--out", default=r"E:\xfmss_data\derived")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for zp in sorted(glob.glob(os.path.join(a.src, "Kelmarsh_SCADA_*.zip"))):
        year = re.search(r"SCADA_(\d{4})_", zp).group(1)
        fd, fe = os.path.join(a.out, f"scada_{year}.pkl"), os.path.join(a.out, f"status_{year}.pkl")
        if os.path.exists(fd) and os.path.exists(fe):
            continue
        z = zipfile.ZipFile(zp)
        frames, events = [], []
        for name in sorted(z.namelist()):
            m = re.match(r"(Turbine_Data|Status)_Kelmarsh_(\d)_", name)
            if not m:
                continue
            kind, wt = m.group(1), int(m.group(2))
            skip, last = header(z, name)
            if kind == "Turbine_Data":
                frames.append(read_turbine(z, name, skip, last, wt))
            else:
                with z.open(name) as f:
                    e = pd.read_csv(f, skiprows=skip, encoding="utf-8", encoding_errors="replace")
                e.columns = [c.lstrip("# ").strip() for c in e.columns]
                e["wt"] = wt
                events.append(e)
        pd.concat(frames, ignore_index=True).to_pickle(fd)
        pd.concat(events, ignore_index=True).to_pickle(fe)
        print("done", year, flush=True)
    df = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(a.out, "scada_*.pkl")))], ignore_index=True)
    df = df.sort_values(["wt", "time"]).drop_duplicates(["wt", "time"])
    ev = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(a.out, "status_*.pkl")))],
                   ignore_index=True).drop_duplicates()
    ev["start"] = pd.to_datetime(ev["Timestamp start"], errors="coerce")
    ev["end"] = pd.to_datetime(ev["Timestamp end"], errors="coerce")
    df.to_pickle(os.path.join(a.out, "kelmarsh_10min.pkl"))
    ev.to_pickle(os.path.join(a.out, "kelmarsh_status.pkl"))
    print(f"rows {len(df):,}  events {len(ev):,}  {df.time.min()} to {df.time.max()}")


if __name__ == "__main__":
    main()
