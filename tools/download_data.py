"""Download the real wind-farm datasets from Zenodo (resumable, MD5-verified).

    python tools/download_data.py [--dest E:\\xfmss_data] [--only kelmarsh|hill_of_towie]

Kelmarsh (CC BY 4.0, doi:10.5281/zenodo.16807551): all files (~3.8 GB).
Hill of Towie (CC BY 4.0, doi:10.5281/zenodo.22662930): documentation, operator-log downtime and the yearly
10-minute SCADA + alarm zips 2016-2026 (~15.3 GB; 2026 = Jan-Apr). Not downloaded: LiDAR, the LiDAR deployment
reports, the 1 Hz turbine fastlog and the sensor photo.
The raw data is kept outside the project folder (default: $XFMSS_DATA, else E:\\xfmss_data) so it never ends up in
the code release.
"""
import argparse, hashlib, json, os, re, sys, time, urllib.request

MAX_TRIES = 40  # per file; each retry resumes the partial file, waiting up to 5 min (Zenodo resets connections at times)

DATASETS = {
    "kelmarsh": {"record": "16807551", "keep": r".*"},
    "hill_of_towie": {"record": "22662930",
                      "keep": r"(README\.md|CHANGELOG\.md|.*description\.csv|.*data_dictionary\.md|.*metadata\.csv|"
                              r".*AeroUp_install_dates\.csv|.*ShutdownDuration\.zip|20(1[6-9]|2[0-6])\.zip)$"},
}


def md5sum(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, path, size):
    """Download url to path, resuming a partial file if the server allows it."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have >= size:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        done = have if mode == "ab" else 0
        t0, last = time.time(), 0.0
        with open(path, mode) as f:
            while True:
                b = r.read(4 << 20)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if time.time() - last > 60:
                    last = time.time()
                    rate = (done - (have if mode == "ab" else 0)) / max(1e-6, last - t0) / 1e6
                    print(f"    {os.path.basename(path)}: {done / 1e6:,.0f} / {size / 1e6:,.0f} MB ({rate:.1f} MB/s)",
                          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.environ.get("XFMSS_DATA", r"E:\xfmss_data"))
    ap.add_argument("--only", choices=list(DATASETS))
    ap.add_argument("--files", default=None, help="regex: download only these file names (for parallel downloads)")
    a = ap.parse_args()
    for name, spec in DATASETS.items():
        if a.only and name != a.only:
            continue
        out = os.path.abspath(os.path.join(a.dest, name))
        os.makedirs(out, exist_ok=True)
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{spec['record']}", timeout=60) as r:
            rec = json.load(r)
        json.dump(rec, open(os.path.join(out, "zenodo_record.json"), "w", encoding="utf-8"), indent=1)
        files = [f for f in rec["files"] if re.fullmatch(spec["keep"], f["key"])
                 and (a.files is None or re.fullmatch(a.files, f["key"]))]
        files.sort(key=lambda f: f["size"])  # small files first: documentation can be read while the rest arrives
        print(f"== {name}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB -> {out}", flush=True)
        for f in files:
            path = os.path.join(out, f["key"])
            want = f["checksum"].split(":", 1)[1]
            ok_flag = path + ".md5ok"
            if os.path.exists(ok_flag) and os.path.exists(path) and os.path.getsize(path) == f["size"]:
                print(f"  ok (already verified) {f['key']}", flush=True)
                continue
            for attempt in range(1, MAX_TRIES + 1):
                try:
                    fetch(f["links"]["self"], path, f["size"])
                    if md5sum(path) == want:
                        open(ok_flag, "w").close()
                        print(f"  ok {f['key']} ({f['size'] / 1e6:,.1f} MB, md5 verified)", flush=True)
                        break
                    print(f"  checksum mismatch for {f['key']}; downloading again", flush=True)
                    os.remove(path)
                except Exception as e:  # network hiccup or Zenodo throttling: wait, then resume the partial file
                    wait = min(30 * attempt, 300)
                    print(f"  retry {attempt}/{MAX_TRIES} for {f['key']} in {wait} s: {type(e).__name__}: {e}", flush=True)
                    time.sleep(wait)
            else:
                print(f"  FAILED {f['key']}", flush=True)
                sys.exit(1)
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
