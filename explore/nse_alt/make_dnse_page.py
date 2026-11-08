#!/usr/bin/env python3
"""Assemble the dendrite-noise-gate page: dnse_template.html + work/dnse_data.json."""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    tpl = open(f"{HERE}/dnse_template.html").read()
    data = json.load(open(f"{HERE}/work/dnse_data.json"))
    sha = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    data["provenance"] = (f"Scripts: <code>explore/nse_alt/build_dnse.py</code> on branch <code>expl/nse-alt</code> at <code>{sha}</code> "
                          f"(main + max_softmax + retina + console and strict-enum fixes); trained groups from <code>run.py</code>, "
                          f"binary <code>build-min/coast</code> (Release, headless, no GPU).")
    page = tpl.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = f"{HERE}/work/dendrite_noise_gate.html"
    open(out, "w").write(page)
    print(out, round(os.path.getsize(out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
