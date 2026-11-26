"""Builds notebooks/phishing_url_detector.ipynb (sectioned, self-contained). Run: python tools/build_notebook.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 🎣 Real-time Phishing URL Detector — raw URL → char-CNN → Chrome extension

The model reads **only the raw URL string** (no HTML, no page features), so it is tiny and runs fully inside the browser.

| Step | What happens |
|---|---|
| Data | UCI **PhiUSIIL** (id 967): 134,850 legitimate + 100,945 phishing URLs |
| Honest evaluation | normalised URLs, **split by registered domain**, shortcut audit + stress test |
| Models | shortcut baselines · TF-IDF + Logistic Regression · **char-CNN (full URL)** · **char-CNN (hostname)** |
| Export | `.pth` · ONNX (checked with onnxruntime) · JS weights for the extension |
| Extension | Chrome MV3, scores every page locally, badge + warning page |

**Kaggle settings:** Internet **On** · GPU T4 (optional, CPU also works) · Add-ons → Google Cloud Services → Cloud Storage
(optional: checkpoints go to your bucket, so a stopped session resumes).  Runtime ≈ 15–25 min on GPU.
""")

md("## 1 · Setup")
code("""
!pip install -q ucimlrepo tldextract onnx onnxscript onnxruntime tabulate 2>&1 | tail -1
import base64, hashlib, json, os, re, shutil, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.precision", 4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)
""")

md("## 2 · Config\nChange only this cell.")
code("""
CFG = dict(
    project_id="mimic-cxr-training",            # your GCP project (bucket for checkpoints)
    bucket="mimic-cxr-training-mimic-ckpt",     # reused private bucket; prefix keeps projects apart
    prefix="phishing-url",
    use_gcs=True,                               # False -> local folder only (no cross-session resume)
    max_len_full=200, max_len_host=64,          # characters fed to the CNN
    epochs=10, batch=512, lr=2e-3, seed=42,
    extension_model="host",                     # model shipped in the extension: "host" or "full"
)
OUT = Path("outputs").resolve()
for d in ("cache", "ckpt", "models", "extension/icons", "results"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
print("outputs ->", OUT)
""")

md("## 3 · Checkpoint store\nGCS bucket when the Kaggle add-on is attached, otherwise a local folder. Every epoch is saved here.")
code("""
def gcs_credentials():
    if "KAGGLE_USER_SECRETS_TOKEN" not in os.environ:
        return None                                              # outside Kaggle: default credentials
    from kaggle_secrets import UserSecretsClient
    if "GCS" in os.environ.get("KAGGLE_KERNEL_INTEGRATIONS", "").upper().split(":"):
        from kaggle_gcp import KaggleKernelCredentials
        from kaggle_secrets import GcpTarget
        return KaggleKernelCredentials(target=GcpTarget.GCS)
    from google.oauth2.credentials import Credentials            # "Google Cloud SDK" add-on
    return Credentials.from_authorized_user_info(json.loads(UserSecretsClient().get_gcloud_credential()))


class Store:
    def __init__(self, cfg):
        self.prefix, self.bucket, self.local = cfg["prefix"], None, OUT / "store"
        if not cfg["use_gcs"]:
            return
        try:
            from google.api_core import exceptions as gexc
            from google.cloud import storage
            client = storage.Client(project=cfg["project_id"], credentials=gcs_credentials())
            b = client.bucket(cfg["bucket"])
            try:
                exists = b.exists()
            except gexc.Forbidden:
                exists = True
            if not exists:
                b.iam_configuration.uniform_bucket_level_access_enabled = True
                b.iam_configuration.public_access_prevention = "enforced"
                b = client.create_bucket(b, location="US")
            self.bucket = b
            print(f"✅ checkpoints -> gs://{cfg['bucket']}/{self.prefix}")
        except Exception as e:
            print(f"⚠️ GCS unavailable ({type(e).__name__}: {str(e)[:150]}) -> local {self.local}")

    def put(self, local, key):
        if self.bucket:
            self.bucket.blob(f"{self.prefix}/{key}").upload_from_filename(str(local))
        else:
            (self.local / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, self.local / key)

    def get(self, key, local) -> bool:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        if self.bucket:
            bl = self.bucket.blob(f"{self.prefix}/{key}")
            if not bl.exists():
                return False
            bl.download_to_filename(str(local))
            return True
        src = self.local / key
        if src.exists() and src.resolve() != Path(local).resolve():
            shutil.copy2(src, local)
        return src.exists()


store = Store(CFG)
""")

md("## 4 · Load PhiUSIIL\nOriginal labels: **1 = legitimate, 0 = phishing**. Here `y = 1` means **phishing** (the positive class).")
code("""
def download_phiusiil():
    try:
        from ucimlrepo import fetch_ucirepo
        return fetch_ucirepo(id=967).data.original
    except Exception as e:                                       # UCI API down -> direct zip
        print("ucimlrepo failed:", type(e).__name__, "-> direct download")
        import io, urllib.request, zipfile
        url = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=120).read()))
        return pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]))


raw = OUT / "cache/phiusiil.parquet"
if not store.get("data/phiusiil.parquet", raw):
    df0 = download_phiusiil()
    lab = next(c for c in df0.columns if c.lower() == "label")
    df0[["URL", lab]].rename(columns={lab: "label"}).to_parquet(raw, index=False)
    store.put(raw, "data/phiusiil.parquet")
df0 = pd.read_parquet(raw)
df0["y"] = (df0["label"] == 0).astype(int)
print(f"{len(df0):,} URLs | phishing {df0.y.sum():,} | legitimate {(1 - df0.y).sum():,}")
df0.sample(5, random_state=1)
""")

md("## 5 · Normalise URLs\nThe **same rules run in the extension** (`extension/model.js`): lowercase, drop scheme, `www.` and trailing `/`.\n"
   "This removes easy tells such as `https://www.` that differ between the two sources.")
code(r'''
SCHEME, WWW = re.compile(r"^[a-z][a-z0-9+.-]*://"), re.compile(r"^www\d*\.")


def norm_url(u: str) -> str:
    s = WWW.sub("", SCHEME.sub("", str(u).strip().lower()))
    return s.rstrip("/")


def host_of(n: str) -> str:
    h = re.split(r"[/?#]", n, maxsplit=1)[0].rsplit("@", 1)[-1]
    if not h.startswith("["):
        h = h.split(":", 1)[0]
    return WWW.sub("", h).rstrip(".")


df0["n"] = df0["URL"].map(norm_url)
df0["host"] = df0["n"].map(host_of)
for u in ["https://www.PayPal.com/", "http://secure-login.paypa1.com.verify-account.xyz/signin?id=7"]:
    print(f"{u:70s} -> {norm_url(u):48s} | host: {host_of(norm_url(u))}")
''')

md("## 6 · Shortcut audit\nIf a trivial rule already separates the classes, a high score proves nothing. "
   "Each single feature below is scored by AUROC (0.5 = useless, 1.0 = perfect separation).")
code("""
aud = pd.DataFrame({
    "y": df0.y,
    "has_path_or_query": df0.n.str.contains(r"[/?#]").astype(int),
    "https_in_raw": df0.URL.str.lower().str.startswith("https").astype(int),
    "www_in_raw": df0.URL.str.lower().str.contains(r"://www\\d*\\.").astype(int),
    "url_length": df0.n.str.len(),
    "host_length": df0.host.str.len(),
    "n_dots_host": df0.host.str.count(r"\\."),
})
rows = []
for c in aud.columns[1:]:
    auc = roc_auc_score(aud.y, aud[c])
    rows.append({"feature": c, "AUROC (alone)": max(auc, 1 - auc),
                 "mean legit": aud.loc[aud.y == 0, c].mean(), "mean phishing": aud.loc[aud.y == 1, c].mean()})
shortcuts = pd.DataFrame(rows).sort_values("AUROC (alone)", ascending=False)
shortcuts
""")
md("> Read this table before trusting any model score. A feature with AUROC near 1 alone means the dataset can be solved "
   "without understanding URLs. Section 12 tests whether the models rely on it.")

md("## 7 · Domain-grouped split\nAll URLs of one registered domain (e.g. `paypal.com`) go to the **same** split, "
   "so the test set has only unseen domains. Exact duplicates are removed; URLs with both labels are dropped.")
code("""
import tldextract

_ext = tldextract.TLDExtract(suffix_list_urls=())                # bundled list, no network


def reg_domain(h):
    e = _ext(h)
    return getattr(e, "top_domain_under_public_suffix", None) or e.registered_domain or h


lab_n = df0.groupby("n").y.nunique()
df = df0[df0.n.map(lab_n) == 1].drop_duplicates("n").copy()
print(f"dedup: {len(df0):,} -> {len(df):,} unique URLs ({(lab_n > 1).sum()} with conflicting labels dropped)")
df["dom"] = df.host.map(reg_domain)
b = df.dom.map(lambda d: int(hashlib.md5(d.encode()).hexdigest(), 16) % 100)
df["split"] = np.select([b < 80, b < 90], ["train", "val"], "test")
print(df.groupby("split").agg(urls=("y", "size"), phishing=("y", "mean"), domains=("dom", "nunique")))

# hostname-level data for the host model (same split assignment)
lab_h = df.groupby(["split", "host"]).y.nunique()
H = df.drop_duplicates(["split", "host"]).set_index(["split", "host"])
H = H[lab_h.reindex(H.index) == 1].reset_index()[["split", "host", "y"]]
print(f"host-level rows: {len(H):,} (hosts with mixed labels dropped)")
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
""")

md("## 8 · Baseline: TF-IDF char n-grams + Logistic Regression")
code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

t0 = time.time()
vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3, max_features=300_000,
                      sublinear_tf=True, dtype=np.float32)
Xtr = vec.fit_transform(tr.n)
lr = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear").fit(Xtr, tr.y)
P = {"tfidf_lr": {s: lr.predict_proba(vec.transform(d.n))[:, 1] for s, d in (("val", va), ("test", te))}}
print(f"TF-IDF + LR done in {time.time() - t0:.0f}s | val AUROC {roc_auc_score(va.y, P['tfidf_lr']['val']):.4f}")
""")

md("## 9 · Char-CNN\nEmbedding (32) → three Conv1d (k = 3/5/7, 128 filters) → global max-pool → 64 → 1. About 90k parameters. "
   "Characters: printable ASCII → ids 2–96, other characters → 1, padding → 0.")
code(r'''
VOCAB = 97
_NONPRINT = re.compile(r"[^\x20-\x7e]")


def encode(texts, L):
    buf = b"".join(_NONPRINT.sub("\x01", t[:L]).encode("ascii").ljust(L, b"\x00") for t in texts)
    a = np.frombuffer(buf, np.uint8).reshape(len(texts), L).astype(np.int64)
    return np.where(a >= 32, a - 30, a)                          # 0 pad, 1 unknown, 2..96 printable


class CharCNN(nn.Module):
    def __init__(self, vocab=VOCAB, emb=32, ch=128, ks=(3, 5, 7), hidden=64, drop=0.25):
        super().__init__()
        self.cfg = dict(emb_dim=emb, channels=ch, kernels=list(ks), hidden=hidden)
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, ch, k) for k in ks)
        self.fc1, self.fc2, self.drop = nn.Linear(ch * len(ks), hidden), nn.Linear(hidden, 1), nn.Dropout(drop)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([F.relu(c(e)).amax(dim=2) for c in self.convs], 1)
        return self.fc2(self.drop(F.relu(self.fc1(self.drop(h))))).squeeze(1)


print(f"{sum(p.numel() for p in CharCNN().parameters()):,} parameters |", encode(["paypal.com"], 16)[0])
''')

md("## 10 · Train (resumable)\nEach epoch saves model + optimizer + scheduler to the store. Re-running after a stop continues "
   "from the last finished epoch. The best epoch is picked by validation AUROC. After a rerun the table still shows "
   "every saved epoch (`source = checkpoint`).")
code("""
def predict(model, X, bs=4096):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(DEVICE)).float()).cpu().numpy())
    return np.concatenate(out)


def train_cnn(name, Xtr, ytr, Xva, yva):
    fp = hashlib.md5(Xtr[:2000].tobytes() + str((len(Xtr), CFG["epochs"], CFG["lr"])).encode()).hexdigest()[:8]
    key, local = f"ckpt/{name}-{fp}.pt", OUT / f"ckpt/{name}.pt"
    model = CharCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    steps = (len(Xtr) + CFG["batch"] - 1) // CFG["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=steps * CFG["epochs"])
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    st = {"epoch": 0, "best": -1.0, "best_state": None, "hist": []}
    if store.get(key, local):
        ck = torch.load(local, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = ck["st"]
        print(f"↻ {name}: loaded checkpoint, {st['epoch']}/{CFG['epochs']} epochs already done"
              + (" -> nothing to retrain" if st["epoch"] >= CFG["epochs"] else " -> continuing"))
    loaded = st["epoch"]
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    for ep in range(st["epoch"] + 1, CFG["epochs"] + 1):
        model.train(); t0, tot = time.time(), 0.0
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(CFG["seed"] + ep))
        for i in range(0, len(Xt), CFG["batch"]):
            idx = perm[i:i + CFG["batch"]]
            xb, yb = Xt[idx].to(DEVICE, non_blocking=True), yt[idx].to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(idx)
        auc = roc_auc_score(yva, predict(model, Xva))
        if auc > st["best"]:
            st["best"], st["best_state"] = auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        st["epoch"] = ep
        st["hist"].append({"epoch": ep, "loss": tot / len(Xt), "val_auroc": auc})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "st": st}, local)
        store.put(local, key)
        print(f"{name} | epoch {ep:2d}/{CFG['epochs']} | loss {tot / len(Xt):.4f} | val AUROC {auc:.4f} | {time.time() - t0:.0f}s")
    model.load_state_dict(st["best_state"])
    h = pd.DataFrame(st["hist"])
    h["source"] = np.where(h.epoch <= loaded, "checkpoint", "this session")
    h["best"] = np.where(h.index == h.val_auroc.idxmax(), "★", "")         # first best epoch = model kept
    print(f"✅ {name}: best val AUROC {st['best']:.4f} | all epochs (★ = model kept):")
    display(h)
    return model.eval(), h


Lf, Lh = CFG["max_len_full"], CFG["max_len_host"]
cnn_full, hist_full = train_cnn("charcnn_full", encode(tr.n, Lf), tr.y.values, encode(va.n, Lf), va.y.values)
Htr, Hva = H[H.split == "train"], H[H.split == "val"]
cnn_host, hist_host = train_cnn("charcnn_host", encode(Htr.host, Lh), Htr.y.values, encode(Hva.host, Lh), Hva.y.values)
""")

md("## 11 · Evaluate on unseen domains\nAll models are scored on the **same URL-level test set** (the host model scores each URL by its "
   "hostname, exactly like the extension). Thresholds are set on validation legit URLs: "
   "**warn** = 1% false-positive rate, **block** = 0.1%.")
code("""
P["charcnn_full"] = {s: predict(cnn_full, encode(d.n, Lf)) for s, d in (("val", va), ("test", te))}
P["charcnn_host"] = {s: predict(cnn_host, encode(d.host, Lh)) for s, d in (("val", va), ("test", te))}
P["rule_url_length"] = {s: d.n.str.len().values / 500 for s, d in (("val", va), ("test", te))}
P["rule_has_path"] = {s: d.n.str.contains(r"[/?#]").values.astype(float) for s, d in (("val", va), ("test", te))}
rng = np.random.default_rng(0)


def thresholds(pv, yv):
    neg = pv[yv == 0]
    return {"warn": float(np.quantile(neg, 0.99)), "block": float(np.quantile(neg, 0.999))}


def evaluate(p, y, th, n_boot=300):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    boots = [roc_auc_score(y[i], p[i]) for i in idx]
    r = {"AUROC": roc_auc_score(y, p), "AUROC 95% CI": f"{np.percentile(boots, 2.5):.4f}–{np.percentile(boots, 97.5):.4f}",
         "AUPRC": average_precision_score(y, p), "F1@0.5": f1_score(y, p >= 0.5)}
    for k, t in th.items():
        r[f"TPR@{k}"], r[f"FPR@{k}"] = (p[y == 1] >= t).mean(), (p[y == 0] >= t).mean()
    return r


TH = {m: thresholds(P[m]["val"], va.y.values) for m in P}
results = pd.DataFrame({m: evaluate(P[m]["test"], te.y.values, TH[m]) for m in P}).T
print(f"test: {len(te):,} URLs from {te.dom.nunique():,} unseen domains | phishing {te.y.mean():.1%}")
results
""")

md("## 12 · Stress test: does the model just look at the path?\n"
   "- **legit + path**: a realistic path is added to every legitimate test URL (`/login`, `/search?q=…`). A model that learned "
   "\"has a path → phishing\" will now raise false alarms on normal browsing.\n"
   "- **phishing → host only**: the path is removed from phishing URLs. Can the model still catch them?")
code("""
PATHS = ["/login", "/index.html", "/search?q=weather+dhaka", "/account/settings", "/news/2025/03/story-1234.html",
         "/products/item?id=42", "/wp-content/uploads/2024/05/report.pdf", "/en-us/help/contact"]
leg, phi = te[te.y == 0], te[te.y == 1]
leg_path = [n + PATHS[i % len(PATHS)] for i, n in enumerate(leg.n)]
phi_host = list(phi.host)
score = {
    "tfidf_lr": lambda xs: lr.predict_proba(vec.transform(xs))[:, 1],
    "charcnn_full": lambda xs: predict(cnn_full, encode(xs, Lf)),
    "charcnn_host": lambda xs: predict(cnn_host, encode([host_of(x) for x in xs], Lh)),
}
stress = pd.DataFrame({m: {
    "FPR legit (original)": (f(list(leg.n)) >= TH[m]["warn"]).mean(),
    "FPR legit + path": (f(leg_path) >= TH[m]["warn"]).mean(),
    "TPR phishing (original)": (f(list(phi.n)) >= TH[m]["warn"]).mean(),
    "TPR phishing host-only": (f(phi_host) >= TH[m]["warn"]).mean(),
} for m, f in score.items()}).T
stress
""")
md("> For a browser extension the row that matters is **FPR legit + path**: every real page you open has a path. "
   "The model with a low value there is the one that is safe to ship (`CFG['extension_model']`).")

md("## 13 · Training curves")
code("""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for h, n in ((hist_full, "full URL"), (hist_host, "hostname")):
    ax[0].plot(h.epoch, h.loss, marker="o", label=n); ax[1].plot(h.epoch, h.val_auroc, marker="o", label=n)
ax[0].set(title="train loss", xlabel="epoch"); ax[1].set(title="val AUROC", xlabel="epoch")
for a in ax:
    a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(OUT / "results/training_curves.png", dpi=130); plt.show()
""")

md("## 14 · Export: `.pth` + ONNX")
code("""
import onnxruntime as ort

for name, m, L in (("charcnn_full", cnn_full, Lf), ("charcnn_host", cnn_host, Lh)):
    m = m.cpu().eval()
    torch.save({"state_dict": m.state_dict(), "config": {**m.cfg, "max_len": L, "vocab": VOCAB},
                "thresholds": TH[name]}, OUT / f"models/{name}.pth")
    x = torch.from_numpy(encode(["paypal.com/signin", "example.org"], L))
    onnx_path = OUT / f"models/{name}.onnx"
    kw = dict(input_names=["ids"], output_names=["logit"], dynamic_axes={"ids": {0: "batch"}, "logit": {0: "batch"}},
              opset_version=17)
    try:
        torch.onnx.export(m, (x,), onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, (x,), onnx_path, **kw)
    ref = m(x).detach().numpy()
    got = ort.InferenceSession(str(onnx_path)).run(None, {"ids": x.numpy()})[0]
    print(f"{name}: .pth + .onnx ({onnx_path.stat().st_size / 1e3:.0f} KB) | max |torch-onnx| = {np.abs(ref - got).max():.2e}")
cnn_full, cnn_host = cnn_full.to(DEVICE), cnn_host.to(DEVICE)
""")

md("## 15 · Chrome extension\nThe files below are written to `outputs/extension/`. The trained weights are added as `weights.js`.")
for f in ("manifest.json", "model.js", "background.js", "popup.html", "popup.js", "warning.html", "warning.js"):
    code(f"%%writefile outputs/extension/{f}\n" + (ROOT / "extension" / f).read_text(encoding="utf-8").rstrip())

md("### Weights, icons and zip")
code("""
from PIL import Image, ImageDraw

ship = CFG["extension_model"]
m, L = (cnn_host, Lh) if ship == "host" else (cnn_full, Lf)
meta = {"input": ship, "max_len": L, **m.cfg, "thresholds": TH[f"charcnn_{ship}"],
        "metrics": {"auroc": float(results.loc[f"charcnn_{ship}", "AUROC"])},
        "weights": {k: base64.b64encode(v.detach().cpu().float().numpy().tobytes()).decode()
                    for k, v in m.state_dict().items()}}
(OUT / "extension/weights.js").write_text("self.PHISH_MODEL = " + json.dumps(meta) + ";\\n")

for s in (16, 48, 128):                                          # simple shield icon
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.polygon([(s * .5, s * .04), (s * .92, s * .2), (s * .85, s * .62), (s * .5, s * .96), (s * .15, s * .62),
               (s * .08, s * .2)], fill=(26, 115, 232))
    d.line([(s * .3, s * .5), (s * .45, s * .66), (s * .72, s * .34)], fill="white", width=max(1, s // 10))
    im.save(OUT / f"extension/icons/{s}.png")
shutil.make_archive(str(OUT / "phishguard-extension"), "zip", OUT / "extension")
print(f"extension uses: charcnn_{ship} | thresholds {TH[f'charcnn_{ship}']}")
print("zip ->", OUT / "phishguard-extension.zip")
""")

md("### Parity check: JavaScript vs PyTorch\nRuns the extension's `model.js` in Node.js on test URLs and compares with PyTorch.")
code("""
import subprocess

sample = list(te.URL.sample(200, random_state=0)) + ["https://www.google.com/search?q=x", "http://paypa1-login.top/verify"]
js = OUT / "results/parity.js"
js.write_text(f'''
globalThis.self = globalThis;
require("{OUT}/extension/weights.js"); require("{OUT}/extension/model.js");
const m = PhishModel.load(self.PHISH_MODEL);
const urls = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(urls.map(u => m.score(u).p)));''')
text = [host_of(norm_url(u)) if ship == "host" else norm_url(u) for u in sample]
ref = predict(m, encode(text, L))
if shutil.which("node"):
    got = np.array(json.loads(subprocess.run(["node", str(js)], input=json.dumps(sample), capture_output=True,
                                             text=True, check=True).stdout))
    print(f"max |JS - PyTorch| over {len(sample)} URLs: {np.abs(got - ref).max():.2e}  (✅ if < 1e-4)")
else:
    print("node not installed here; parity is covered by tests/test_parity.py in the repo")
""")

md("## 16 · Try it")
code("""
def check(url):
    n = norm_url(url)
    x = host_of(n) if ship == "host" else n
    p = float(predict(m, encode([x], L))[0])
    t = TH[f"charcnn_{ship}"]
    v = "🔴 phishing" if p >= t["block"] else "🟠 suspicious" if p >= t["warn"] else "🟢 safe"
    return {"url": url, "model input": x, "p(phishing)": round(p, 4), "verdict": v}


pd.DataFrame(map(check, [
    "https://www.google.com/search?q=dhaka+weather", "https://github.com/login", "https://www.bkash.com/",
    "http://bkash-verify-account.secure-update.xyz/login", "http://paypal.com.account-review.info/signin",
    "https://drive-google-docs.web.app/file/d/7Hk2", "http://192.168.10.5/bank/login.php"]))
""")

md("## 17 · Save results + zip for GitHub\nWrites the result files, rebuilds the extension zip and packs everything the repo needs "
   "into **`results_for_github.zip`** (Output tab → download → unzip into the repo).")
code("""
import zipfile

summary = {"data": {"urls_after_dedup": len(df), "test_urls": len(te), "test_domains": int(te.dom.nunique())},
           "shortcuts": shortcuts.to_dict("records"), "results": results.reset_index(names="model").to_dict("records"),
           "stress": stress.reset_index(names="model").to_dict("records"), "thresholds": TH,
           "extension_model": f"charcnn_{ship}", "config": CFG}
(OUT / "results/results.json").write_text(json.dumps(summary, indent=2, default=str))
with open(OUT / "results/RESULTS.md", "w") as f:
    f.write("## Shortcut audit\\n\\n" + shortcuts.to_markdown(index=False, floatfmt=".4f"))
    f.write("\\n\\n## Test set (unseen registered domains)\\n\\n" + results.to_markdown(floatfmt=".4f"))
    f.write("\\n\\n## Stress test (warn threshold)\\n\\n" + stress.to_markdown(floatfmt=".4f") + "\\n")
(OUT / "results/parity.js").unlink(missing_ok=True)              # helper script with local paths, not for git

# 1) Chrome extension zip (load-unpacked folder, zipped)
need = ["manifest.json", "model.js", "background.js", "popup.html", "popup.js", "warning.html", "warning.js",
        "weights.js", "icons/16.png", "icons/48.png", "icons/128.png"]
missing = [f for f in need if not (OUT / "extension" / f).exists()]
assert not missing, f"extension files missing: {missing} -> run section 15 first"
ext_zip = Path(shutil.make_archive(str(OUT / "phishguard-extension"), "zip", OUT / "extension"))

# 2) everything for the GitHub repo in one zip (same layout as the repo)
gh = Path.cwd() / "results_for_github.zip"
with zipfile.ZipFile(gh, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("models", "results", "extension"):
        for p in sorted((OUT / sub).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    z.write(ext_zip, ext_zip.name)
with zipfile.ZipFile(gh) as z:
    assert z.testzip() is None, "zip is corrupt"
    names = z.namelist()

for p in [OUT / "results/results.json", OUT / "results/RESULTS.md", ext_zip, gh] + sorted((OUT / "models").iterdir()):
    store.put(p, f"final/{p.name}")
print(f"✅ {gh.name}: {len(names)} files, {gh.stat().st_size / 1e6:.1f} MB  (also in the bucket under final/)")
print(f"✅ {ext_zip.name}: {ext_zip.stat().st_size / 1e3:.0f} KB")
for folder in ("models", "results", "extension"):
    print(f"   {folder + '/':11s}", ", ".join(n.split("/", 1)[1] for n in names if n.startswith(folder + "/")))
print("Unzip results_for_github.zip into the repo folder (it fills models/, results/, extension/).")
""")

def main():
    nb = nbf.v4.new_notebook()
    nb.cells = C
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks/phishing_url_detector.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print("wrote", out, f"({len(C)} cells)")


if __name__ == "__main__":
    main()
