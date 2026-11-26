"""JS extension (model.js) must give the same score as the notebook's PyTorch model."""
import base64, json, re, shutil, subprocess
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/phishing_url_detector.ipynb"


def _ns():
    ns = {}
    exec("import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F", ns)
    for c in nbformat.read(NB, 4).cells:
        if c.cell_type == "code" and ("def norm_url" in c.source or "class CharCNN" in c.source):
            src = re.split(r"\n(?=df0\[|for u in|print\()", c.source)[0]     # definitions only
            exec(src, ns)
    return ns


URLS = ["https://www.PayPal.com/", "http://user@www2.bank.com.bd:8080/login?x=1#top", "HTTPS://Example.ORG//",
        "http://192.168.0.1/a.php", "https://বাংলা.com/পাতা", "bkash-verify.secure.xyz/" + "a" * 300, "  x.com.  "]


def test_normalisation_and_encoding_match_js(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    ns = _ns()
    js = tmp_path / "t.js"
    js.write_text(f"""globalThis.self = globalThis; require("{ROOT}/extension/model.js");
const u = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(u.map(x => {{ const n = PhishModel.normUrl(x);
  return [n, PhishModel.hostOf(n), Array.from(PhishModel.encode(n, 200))]; }})));""")
    got = json.loads(subprocess.run(["node", js], input=json.dumps(URLS), capture_output=True, text=True,
                                    check=True).stdout)
    for u, (n, h, ids) in zip(URLS, got):
        assert n == ns["norm_url"](u) and h == ns["host_of"](n)
        assert ids == ns["encode"]([n], 200)[0].tolist()


def test_forward_matches_torch(tmp_path):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    ns = _ns()
    torch.manual_seed(0)
    m = ns["CharCNN"]().eval()
    for p in m.parameters():
        p.data.normal_(0, 0.3)
    L = 64
    meta = {"input": "host", "max_len": L, **m.cfg, "thresholds": {"warn": .5, "block": .9}, "metrics": {"auroc": 1},
            "weights": {k: base64.b64encode(v.float().numpy().tobytes()).decode() for k, v in m.state_dict().items()}}
    (tmp_path / "w.js").write_text("self.PHISH_MODEL = " + json.dumps(meta) + ";")
    js = tmp_path / "t.js"
    js.write_text(f"""globalThis.self = globalThis; require("{tmp_path}/w.js"); require("{ROOT}/extension/model.js");
const m = PhishModel.load(self.PHISH_MODEL);
console.log(JSON.stringify(JSON.parse(require("fs").readFileSync(0, "utf8")).map(u => m.score(u).p)));""")
    got = np.array(json.loads(subprocess.run(["node", js], input=json.dumps(URLS), capture_output=True, text=True,
                                             check=True).stdout))
    hosts = [ns["host_of"](ns["norm_url"](u)) for u in URLS]
    with torch.no_grad():
        ref = torch.sigmoid(m(torch.from_numpy(ns["encode"](hosts, L)))).numpy()
    assert np.abs(got - ref).max() < 1e-5
