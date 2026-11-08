#!/usr/bin/env python3
"""Assemble the max-softmax page: msm_template.html + work/msm_data.json + msm_findings.html."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tpl = open(f"{HERE}/msm_template.html").read()
    data = json.load(open(f"{HERE}/work/msm_data.json"))
    sha = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    data["provenance"] = (f"Scripts: <code>explore/nse_alt/</code> (run.py --train-inh, build_msm.py) on branch "
                          f"<code>expl/nse-alt</code> at <code>{sha}</code> plus the uncommitted max_softmax inhibitor; "
                          f"binary <code>build-min/coast</code> (Release, headless, no GPU).")
    findings = open(f"{HERE}/msm_findings.html").read() if os.path.exists(f"{HERE}/msm_findings.html") else ""
    page = (tpl.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
               .replace("<!--__FINDINGS__-->", findings))
    out = f"{HERE}/work/max_softmax.html"
    open(out, "w").write(page)
    print(out, round(os.path.getsize(out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
