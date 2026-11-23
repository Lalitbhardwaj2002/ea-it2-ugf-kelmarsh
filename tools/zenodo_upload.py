"""Upload the replication package to Zenodo as a DRAFT. Nothing is published unless you add --publish.

    set ZENODO_TOKEN=<your personal access token>        (PowerShell: $env:ZENODO_TOKEN = "<token>"; Git Bash: export ZENODO_TOKEN=<token>)
    python tools/zenodo_upload.py                         creates a draft, reserves its DOI, uploads the zip, prints both
    python tools/zenodo_upload.py --sandbox               the same on sandbox.zenodo.org (a test site; needs a sandbox token)
    python tools/zenodo_upload.py --publish <id>          publishes the draft with that id (cannot be undone)

Create the token at https://zenodo.org/account/settings/applications/tokens/new/ with the scopes deposit:write and deposit:actions.
The token is read from the environment only; it is never written to a file. The metadata come from .zenodo.json, and the zip holds
the package files only (no hidden folders, caches or generated figures). Run tools/make_release.py first so the folder is clean.
Use one route per version: if the GitHub repository is switched on in Zenodo, a GitHub release already makes the record and its DOI.
Only the standard library is used.
"""
import argparse, io, json, os, sys, urllib.error, urllib.request, zipfile

HERE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def call(method, url, token, data=None, ctype="application/json"):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    req = urllib.request.Request(url, data=body, method=method, headers={"Authorization": f"Bearer {token}", "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo answered {e.code}: {e.read().decode(errors='ignore')[:600]}")


def package_zip():
    buf, n = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for base, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"
                       and os.path.relpath(os.path.join(base, d), HERE).replace(os.sep, "/") not in ("paper/figures_sncs",)]
            for f in sorted(files):
                if f.endswith(".zip"):
                    continue
                p = os.path.join(base, f)
                z.write(p, os.path.join("ea-it2-ugf-kelmarsh", os.path.relpath(p, HERE)))
                n += 1
    return buf.getvalue(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ap.add_argument("--publish", metavar="ID", help="publish an existing draft (cannot be undone)")
    a = ap.parse_args()
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN first (see the top of this file).")
    api = "https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api"
    if a.publish:
        r = call("POST", f"{api}/deposit/depositions/{a.publish}/actions/publish", token)
        print("published:", r.get("doi"), r.get("links", {}).get("html"))
        return
    meta = json.load(open(os.path.join(HERE, ".zenodo.json"), encoding="utf-8"))
    meta["prereserve_doi"] = True
    data, n = package_zip()
    dep = call("POST", f"{api}/deposit/depositions", token, {"metadata": meta})
    print(f"Draft {dep['id']} created.")
    name = f"ea-it2-ugf-kelmarsh-v{meta.get('version', '1.0.0')}.zip"
    call("PUT", f"{dep['links']['bucket']}/{name}", token, data, "application/octet-stream")
    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"Uploaded {name} ({n} files, {len(data) / 1e6:.1f} MB).")
    print(f"Reserved DOI: {doi}   (put it in the manuscript's Data and Code availability statements)")
    print(f"Check it here: {dep['links']['html']}")
    print(f"Publish when ready: python tools/zenodo_upload.py{' --sandbox' if a.sandbox else ''} --publish {dep['id']}")


if __name__ == "__main__":
    main()
