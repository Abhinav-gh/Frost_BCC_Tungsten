"""Search the public places a method's code, checkpoints or data could be, and write a dated markdown log.

Part of the unreleased-baseline skill (step 1). Only public, unauthenticated APIs; every query and its answer, including
"nothing", goes into the log, so the search is a record a reviewer can check, not a sentence in the paper.

  python .claude/skills/unreleased-baseline/artifact_hunt.py --name BoostMD --arxiv 2412.18633 \\
      --authors LarsSchaaf --orgs instadeepai --libs ACEsuit/mace --terms boost --out log.md

Sources: GitHub (repository search, authors' and organisations' repositories, branches of the libraries the method
builds on), Hugging Face (models, datasets, the paper page's githubRepo), Zenodo, arXiv HTML (availability statements,
links), OpenAlex (papers citing it, whose code may reimplement it). Rate limits: GitHub allows 10 searches a minute
unauthenticated; the script waits between calls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "frost-artifact-hunt", "Accept": "application/vnd.github+json"}


def get(url: str, raw: bool = False, timeout: int = 30):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
        return body if raw else json.loads(body)
    except Exception as e:  # every failure is logged as such
        return {"error": f"{type(e).__name__}: {e}"}


def q(s: str) -> str:
    return urllib.parse.quote(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="method name as people write it")
    ap.add_argument("--arxiv", default="", help="arXiv id")
    ap.add_argument("--title", default="", help="paper title (citing-paper search when no arXiv id)")
    ap.add_argument("--authors", default="", help="comma-separated GitHub handles of the authors")
    ap.add_argument("--orgs", default="", help="comma-separated GitHub organisations (labs, companies)")
    ap.add_argument("--libs", default="", help="comma-separated owner/repo the method builds on (branches are checked)")
    ap.add_argument("--terms", default="", help="extra comma-separated keywords to match in names and branches")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    terms = [t.lower() for t in [a.name, *a.terms.split(",")] if t]
    hit = lambda s: any(t in (s or "").lower() for t in terms)  # noqa: E731
    log = [f"# Artifact hunt: {a.name}", "", f"Run {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC.", ""]

    def section(title, lines):
        log.extend([f"## {title}", "", *(lines or ["- nothing found"]), ""])

    # GitHub repository search
    lines = []
    for term in dict.fromkeys([a.name, *a.terms.split(",")]):
        if not term:
            continue
        d = get(f"https://api.github.com/search/repositories?q={q(term)}&per_page=10")
        if "error" in d or "items" not in d:
            lines.append(f"- search `{term}`: {d.get('error') or d.get('message')}")
        else:
            names = [f"[{r['full_name']}]({r['html_url']})" for r in d["items"]]
            lines.append(f"- search `{term}`: {d['total_count']} repos" + (": " + ", ".join(names) if names else ""))
        time.sleep(7)
    section("GitHub repository search", lines)

    # authors' and organisations' repositories
    lines = []
    for who in [x for x in (a.authors + "," + a.orgs).split(",") if x]:
        d = get(f"https://api.github.com/users/{who}/repos?per_page=100&sort=updated")
        if isinstance(d, dict):
            lines.append(f"- `{who}`: {d.get('error') or d.get('message')}")
            continue
        rel = [f"[{r['name']}]({r['html_url']})" for r in d if hit(r["name"]) or hit(r.get("description"))]
        lines.append(f"- `{who}`: {len(d)} repos; matching: {', '.join(rel) if rel else 'none'}")
        time.sleep(1)
    section("Authors' and organisations' repositories", lines)

    # branches of the libraries it builds on
    lines = []
    for lib in [x for x in a.libs.split(",") if x]:
        d = get(f"https://api.github.com/repos/{lib}/branches?per_page=100")
        if isinstance(d, dict):
            lines.append(f"- `{lib}`: {d.get('error') or d.get('message')}")
            continue
        names = [b["name"] for b in d]
        rel = [n for n in names if hit(n)]
        lines.append(f"- `{lib}`: {len(names)} branches; matching: {', '.join(rel) if rel else 'none'}")
        time.sleep(1)
    section("Branches of the base libraries", lines)

    # Hugging Face
    lines = []
    for kind in ("models", "datasets", "spaces"):
        d = get(f"https://huggingface.co/api/{kind}?search={q(a.name)}&limit=20")
        ids = [x["id"] for x in d] if isinstance(d, list) else []
        lines.append(f"- {kind} `{a.name}`: {', '.join(ids) if ids else 'none'}")
    if a.arxiv:
        d = get(f"https://huggingface.co/api/papers/{a.arxiv}")
        lines.append(f"- paper page {a.arxiv}: githubRepo = {d.get('githubRepo') if isinstance(d, dict) else d}")
    section("Hugging Face", lines)

    # Zenodo
    lines = []
    for term in dict.fromkeys([a.name, a.arxiv, a.title]):
        if not term:
            continue
        d = get(f"https://zenodo.org/api/records?q={q(chr(34) + term + chr(34))}&size=10")
        hits_ = d.get("hits", {}).get("hits", []) if isinstance(d, dict) else []
        lines.append(f"- `{term}`: " + (", ".join(f"[{h['metadata']['title'][:70]}]({h['links']['self_html']})"
                                               for h in hits_) or "none"))
    section("Zenodo", lines)

    # arXiv availability statements and links
    lines = []
    if a.arxiv:
        page = get(f"https://arxiv.org/html/{a.arxiv}", raw=True, timeout=60)
        if isinstance(page, dict):
            lines.append(f"- arXiv HTML: {page['error']}")
        else:
            text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
            for k in ("availability", "github.com", "zenodo", "huggingface.co", "upon request", "will be released"):
                for m in re.finditer(re.escape(k), text, flags=re.I):
                    lines.append(f"- `{k}`: ...{text[max(0, m.start() - 120):m.end() + 160]}...")
                    break
            if not lines:
                lines.append("- no availability statement or code link in the HTML")
    section("arXiv availability statements", lines)

    # citing papers (OpenAlex)
    lines = []
    probe = a.title or a.name
    d = get(f"https://api.openalex.org/works?search={q(probe)}&per-page=1")
    res = d.get("results", []) if isinstance(d, dict) else []
    if res:
        wid = res[0]["id"].rsplit("/", 1)[-1]
        lines.append(f"- matched work: {res[0].get('title')} ({wid})")
        c = get(f"https://api.openalex.org/works?filter=cites:{wid}&per-page=100&select=title,publication_year,doi")
        for w in c.get("results", []) if isinstance(c, dict) else []:
            lines.append(f"- cited by ({w.get('publication_year')}): {w.get('title')} {w.get('doi') or ''}")
    section("Citing papers (check each one's code for a reimplementation)", lines)

    with open(a.out, "w") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
