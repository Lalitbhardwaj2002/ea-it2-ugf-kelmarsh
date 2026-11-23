"""Build kaggle_kelmarsh.ipynb: the one-click LLM run of the Kelmarsh study, with the current code embedded.

    python tools/build_kelmarsh_notebook.py
"""
import base64, io, json, os, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = ["run_experiment.py", "requirements.txt", "data/kelmarsh.json"] + [
    f"xfmss/{f}" for f in sorted(os.listdir(os.path.join(ROOT, "xfmss"))) if f.endswith(".py")]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(os.path.join(ROOT, f), f)
code_zip = base64.b64encode(buf.getvalue()).decode()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("""# Kelmarsh LLM run

1. In the right panel, open **Settings** and set **Accelerator = GPU T4 x2** and **Internet = On**.
2. Click **Save Version**, choose **Save & Run All (Commit)**, then click Save. You can close the browser.
3. When it finishes, open the version, go to the **Output** tab, download `kelmarsh_results.zip` and unzip it in the package folder (it holds `results/` and `logs/`).

It rates 552 turbine-months in 3 wordings each (1,656 ratings) with Phi-3.5-mini. The 1,278 ratings from January 2019 on are also explained with exact
Shapley values and the LLM's own rationale; the first two years only train the link and need the rating alone. It needs more than one Kaggle session: it stops by itself at
11 hours (the archived run had 1,562 of 1,656 ratings by then), and a second run with the first output attached finishes it. Every finished rating is saved as it goes.

If a run is cut short: add its `kelmarsh_results.zip` to the notebook as a dataset (**Add Input**) and run again. Finished ratings are reused.
"""),
    code("""# 1. SETTINGS
MODEL, DTYPE = "microsoft/Phi-3.5-mini-instruct", "float16"
METHODS = "kernelshap_exact,self_rationale"
EXPLAIN_FROM = "kelmarsh_2019-01"   # earlier months only train the link: rating without explanations
RATINGS = 1656
STOP_AFTER_HOURS = 11.0
import os
os.environ["XFMSS_RUBRIC"] = "fleet"   # scale guide written for fleet medians
"""),
    code("""# 2. install
!pip install -q "inseq==0.7.1" "shap>=0.48" "transformers>=4.56" accelerate
"""),
    code(f"""# 3. unpack the embedded code
import base64, io, zipfile, os
CODE_ZIP = \"\"\"{code_zip}\"\"\"
os.makedirs('/kaggle/working/xfmss', exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall('/kaggle/working/xfmss')
%cd /kaggle/working/xfmss
"""),
    code("""# 4. GPU check
import torch
assert torch.cuda.is_available(), "No GPU! Settings -> Accelerator -> GPU T4 x2"
N_GPU = torch.cuda.device_count()
print(N_GPU, "x", torch.cuda.get_device_name(0))
"""),
    code("""# 5. run (one process per GPU; each takes every 2nd rating; results are written row by row)
import subprocess, time, glob, shutil
os.makedirs("results", exist_ok=True); os.makedirs("logs", exist_ok=True)
for f in glob.glob("/kaggle/input/**/kelmarsh_shard*.jsonl", recursive=True):
    shutil.copy(f, "results/"); print("continuing from", f)
n = max(1, N_GPU)
procs = []
for g in range(n):
    cmd = (f"CUDA_VISIBLE_DEVICES={g} python run_experiment.py --model {MODEL} --dtype {DTYPE} --bench data/kelmarsh.json "
           f"--methods {METHODS} --explain_from {EXPLAIN_FROM} --variants 3 --shard {g}/{n} --out results/kelmarsh_shard{g}.jsonl > logs/gpu{g}.log 2>&1")
    procs.append(subprocess.Popen(cmd, shell=True))
    if g < n - 1:
        time.sleep(180)  # stagger model loading so two copies are never in CPU memory at once
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
    if time.time() - t0 > STOP_AFTER_HOURS * 3600:
        print("Time limit reached: stopping; finished rows are kept.")
        for p in procs:
            p.terminate()
        break
    if int(time.time() - t0) % 600 < 60:
        print(f"{(time.time() - t0) / 60:.0f} min: {sum(sum(1 for _ in open(f)) for f in glob.glob('results/*.jsonl'))} of {RATINGS} ratings done", flush=True)
for f in sorted(glob.glob("logs/*.log")):
    print("==", f); print("".join(open(f).readlines()[-4:]))
"""),
    code("""# 6. package (download from the Output tab, or the link below)
import base64, zipfile
from IPython.display import HTML, display
zp = "/kaggle/working/kelmarsh_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ["results/*.jsonl", "logs/*.log"]:
        for f in glob.glob(pat):
            z.write(f)
rows = sum(sum(1 for _ in open(f)) for f in glob.glob("results/*.jsonl"))
print("ratings saved:", rows, "of", RATINGS)
b64 = base64.b64encode(open(zp, "rb").read()).decode()
display(HTML(f'<a download="kelmarsh_results.zip" href="data:application/zip;base64,{b64}" style="font-size:16px;font-weight:bold">Download kelmarsh_results.zip</a>'))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(ROOT, "kaggle_kelmarsh.ipynb"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_kelmarsh.ipynb", f"({len(FILES)} files embedded, {len(code_zip) // 1024} KB)")
