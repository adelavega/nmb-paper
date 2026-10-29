#!/usr/bin/env python3
"""Fetch citation and author details for the benchmark's source meta-analyses.

Used to work out who to contact for data-redistribution permission, and to build the
acknowledgement/citation list for the paper. Writes results/source_citations.{json,md}.
Set NCBI_EMAIL to the contact address NCBI asks for.

PubMed does not mark which author is corresponding, so this reports the signals that exist:
every author with their affiliation, any e-mail address appearing in an affiliation (which in
practice is the corresponding author's), and the first/last author. Confirm against the article
itself before writing to anyone.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parent.parent          # nmb-paper
# the NeuroMetaBench dataset repo: data/, analysis/ reference maps, and its scripts
# (nimads_meta, map_mask); override with NMB_ROOT
NMB = Path(__import__("os").environ.get("NMB_ROOT", REPO.parent / "neurometabench")).resolve()
__import__("sys").path.append(str(NMB / "scripts"))
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
CONTACT = __import__("os").environ.get("NCBI_EMAIL", "")   # NCBI asks for a contact (E-utilities policy)

# project -> source meta-analysis PMID, from data/meta_datasets.csv and raw/meta-datasets/
PROJECTS = {
    "social": "36436737",
    "vbm_of_ptsd": "36100907",
    "vbm_of_substance_use": "36115222",
    "dementia": "35664889",
    "cue_reactivity": "34400176",
    "emotion_regulation_2022": "35413444",
    "decision_making": "32078973",
    "problem_solving": "29944961",
    "executive_function": "22282036",
}

# Affiliation strings usually end in a full stop, so trim trailing punctuation.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w-]+(?:\.[\w-]+)*")


def fetch(pmids: list[str]) -> ET.Element:
    q = urllib.parse.urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        "tool": "neurometabench", "email": CONTACT,
    })
    req = urllib.request.Request(f"{EUTILS}?{q}",
                                 headers={"User-Agent": f"neurometabench ({CONTACT})"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return ET.fromstring(r.read())


def text(node, path, default=""):
    el = node.find(path)
    return "".join(el.itertext()).strip() if el is not None else default


def parse_article(art: ET.Element) -> dict:
    cit = art.find("MedlineCitation")
    a = cit.find("Article")
    pmid = text(cit, "PMID")

    authors, emails = [], []
    for au in a.findall("AuthorList/Author"):
        last, fore = text(au, "LastName"), text(au, "ForeName")
        coll = text(au, "CollectiveName")
        name = coll or f"{fore} {last}".strip()
        affs = [text(x, ".") for x in au.findall("AffiliationInfo/Affiliation")]
        for af in affs:
            emails += EMAIL_RE.findall(af)
        authors.append({"name": name, "last": last, "affiliations": affs})

    ids = {e.get("IdType"): (e.text or "").strip()
           for e in art.findall("PubmedData/ArticleIdList/ArticleId")}
    year = (text(a, "Journal/JournalIssue/PubDate/Year")
            or text(a, "Journal/JournalIssue/PubDate/MedlineDate")[:4]
            or text(cit, "DateCompleted/Year"))

    return {
        "pmid": pmid,
        "title": text(a, "ArticleTitle"),
        "journal": text(a, "Journal/Title"),
        "journal_abbrev": text(a, "Journal/ISOAbbreviation"),
        "year": year,
        "volume": text(a, "Journal/JournalIssue/Volume"),
        "issue": text(a, "Journal/JournalIssue/Issue"),
        "pages": text(a, "Pagination/MedlinePgn"),
        "doi": ids.get("doi", ""),
        "pmc": ids.get("pmc", ""),
        "authors": authors,
        "emails_in_affiliations": sorted(set(emails)),
        "first_author": authors[0]["name"] if authors else "",
        "last_author": authors[-1]["name"] if authors else "",
    }


def citation(rec: dict) -> str:
    names = []
    for au in rec["authors"]:
        if au["last"]:
            initials = "".join(p[0] for p in au["name"].split()[:-1] if p)
            names.append(f"{au['last']} {initials}")
        else:
            names.append(au["name"])
    auth = ", ".join(names)
    bits = [f"{auth}. {rec['title']}"]
    j = rec["journal_abbrev"] or rec["journal"]
    tail = f" {j}. {rec['year']}"
    if rec["volume"]:
        tail += f";{rec['volume']}"
    if rec["issue"]:
        tail += f"({rec['issue']})"
    if rec["pages"]:
        tail += f":{rec['pages']}"
    bits.append(tail + ".")
    if rec["doi"]:
        bits.append(f" doi:{rec['doi']}.")
    bits.append(f" PMID:{rec['pmid']}.")
    return "".join(bits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results")
    args = ap.parse_args(argv)
    if not CONTACT:
        ap.error("set NCBI_EMAIL to a contact address; NCBI E-utilities asks for one")

    root = fetch(list(PROJECTS.values()))
    by_pmid = {}
    for art in root.findall("PubmedArticle"):
        rec = parse_article(art)
        by_pmid[rec["pmid"]] = rec

    missing = [p for p in PROJECTS.values() if p not in by_pmid]
    if missing:
        print(f"WARNING: no record returned for {missing}", file=sys.stderr)

    out = []
    for proj, pmid in PROJECTS.items():
        rec = by_pmid.get(pmid)
        if rec is None:
            continue
        rec = dict(rec, project=proj, citation=citation(rec))
        out.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "source_citations.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# Source meta-analyses of NeuroMetaBench",
             "",
             "The nine expert meta-analyses whose curation records the benchmark redistributes.",
             "Generated by `scripts/fetch_source_citations.py` from PubMed.",
             "",
             "PubMed does not flag the corresponding author. The e-mail column is any address",
             "appearing in an author affiliation, which is in practice the corresponding author's;",
             "**confirm against the article before writing to anyone.**",
             ""]
    for rec in out:
        lines += [f"## {rec['project']}", "",
                  f"**{rec['citation']}**", "",
                  f"- First author: {rec['first_author']}",
                  f"- Last author: {rec['last_author']}",
                  f"- Contact e-mail(s) found: "
                  f"{', '.join(rec['emails_in_affiliations']) or '— none in PubMed —'}",
                  f"- Authors ({len(rec['authors'])}): "
                  + "; ".join(a["name"] for a in rec["authors"]),
                  ""]
        if rec["authors"] and rec["authors"][-1]["affiliations"]:
            lines += [f"- Last-author affiliation: {rec['authors'][-1]['affiliations'][0]}", ""]
    (args.out_dir / "source_citations.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {args.out_dir / 'source_citations.json'}")
    print(f"wrote {args.out_dir / 'source_citations.md'}")
    print(f"\n{len(out)}/{len(PROJECTS)} source meta-analyses resolved")
    for rec in out:
        em = ", ".join(rec["emails_in_affiliations"]) or "(no e-mail in PubMed)"
        print(f"  {rec['project']:<26}{rec['pmid']:<10}{rec['last_author'][:24]:<26}{em}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
