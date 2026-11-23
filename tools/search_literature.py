"""Reproducible literature search behind the novelty statement of the article.

    python tools/search_literature.py

Searches titles and abstracts in OpenAlex, and all fields in arXiv, for each pair of phrases, and writes every record found to
paper/literature_search.csv for screening. The run date and counts are reported in the article (Research Gaps and Objectives).
"""
import csv, datetime, os, re, sys, time, urllib.parse, urllib.request
import json

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
UA = {"User-Agent": "xfmss-literature-search/1.0"}
LLM = ['"large language model"', '"large language models"', '"LLM"', '"GPT"']
FUZZY = ['"interval type-2"', '"type-2 fuzzy"']
UGF = ['"universal generating function"']
PAIRS = [("IT2 with UGF", FUZZY, UGF), ("LLM with UGF", LLM, UGF), ("LLM with IT2", LLM, FUZZY),
         ("LLM with multi-state reliability", LLM, ['"multi-state system"', '"multistate system"']),
         ("Explanation screening with wind reliability", ['"shapley"', '"attribution"', '"explanation"'], ['"wind turbine" "large language model"', '"wind farm" "large language model"'])]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return r.read().decode("utf-8", "replace")


def openalex(a, b):
    f = f"title_and_abstract.search:{a},title_and_abstract.search:{b}"
    q = urllib.parse.urlencode({"filter": f, "per-page": 50, "select": "id,title,publication_year,doi,primary_location"})
    d = json.loads(get("https://api.openalex.org/works?" + q))
    return d["meta"]["count"], [(w.get("title") or "", w.get("publication_year"), w.get("doi") or w["id"]) for w in d["results"]]


def arxiv(a, b):
    q = urllib.parse.urlencode({"search_query": f"all:{a} AND all:{b}", "max_results": 50})
    t = get("http://export.arxiv.org/api/query?" + q)
    entries = re.findall(r"<entry>(.*?)</entry>", t, re.S)
    out = []
    for e in entries:
        ti = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
        out.append((ti, re.search(r"<published>(\d{4})", e).group(1), re.search(r"<id>(.*?)</id>", e).group(1)))
    return len(out), out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows, seen = [], set()
    print("search date", datetime.date.today())
    for name, A, B in PAIRS:
        n_oa, n_ax, new = 0, 0, 0
        for a, b in [(a, b) for a in A for b in B]:
            for src, fn in (("OpenAlex", openalex), ("arXiv", arxiv)):
                if src == "arXiv" and " " in b.replace('"', "", 2).strip('"') and '" "' in b:
                    continue
                try:
                    n, recs = fn(a, b)
                except Exception as e:
                    print(f"  {src} failed for {a} + {b}: {type(e).__name__}")
                    continue
                if src == "OpenAlex":
                    n_oa += n
                else:
                    n_ax += n
                for ti, y, ident in recs:
                    key = re.sub(r"\W+", "", ti.lower())
                    if key and key not in seen:
                        seen.add(key)
                        new += 1
                        rows.append({"topic": name, "source": src, "year": y, "title": ti, "id": ident})
                time.sleep(3.2 if src == "arXiv" else 0.3)
        print(f"{name}: OpenAlex {n_oa}, arXiv {n_ax}, unique new records {new}")
    with open(os.path.join(ROOT, "paper", "literature_search.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["topic", "source", "year", "title", "id"])
        w.writeheader()
        w.writerows(rows)
    print("unique records", len(rows))
    for r in rows:
        print(f"[{r['topic'][:18]:18s}] {r['year']} {r['title'][:118]}")


if __name__ == "__main__":
    main()
