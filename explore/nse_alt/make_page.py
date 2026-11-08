#!/usr/bin/env python3
"""Assemble the explorer page: template + data + measures.js + findings."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tpl = open(f"{HERE}/explorer_template.html").read()
    data = json.load(open(f"{HERE}/work/explorer_data.json"))
    sha = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    data["provenance"] = (f"Scripts: <code>explore/nse_alt/</code> (run.py, build_data.py, measures.py) on branch "
                          f"<code>expl/nse-alt</code> at <code>{sha}</code>; binary <code>build-min/coast</code> "
                          f"(Release, headless, no GPU). Datasets on this page: {', '.join(data['datasets'])}.")
    findings = open(f"{HERE}/findings.html").read() if os.path.exists(f"{HERE}/findings.html") else ""
    js = open(f"{HERE}/measures.js").read()
    page = (tpl.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
               .replace("/*__MEASURES_JS__*/", js)
               .replace("<!--__FINDINGS__-->", findings))
    out = f"{HERE}/work/beyond_nse.html"
    open(out, "w").write(page)
    print(out, round(os.path.getsize(out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
