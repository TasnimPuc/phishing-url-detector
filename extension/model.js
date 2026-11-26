// Char-CNN phishing scorer. Pure JavaScript, no dependencies, runs inside the service worker.
// Normalisation and encoding mirror the Python notebook exactly (tested for parity).
(function (g) {
  "use strict";
  const SCHEME = /^[a-z][a-z0-9+.-]*:\/\//;
  const WWW = /^www\d*\./;

  function normUrl(u) {
    let s = String(u).trim().toLowerCase();
    s = s.replace(SCHEME, "").replace(WWW, "");
    return s.replace(/\/+$/, "");
  }

  function hostOf(n) {
    let h = n.split(/[\/?#]/, 1)[0];
    const at = h.lastIndexOf("@");
    if (at >= 0) h = h.slice(at + 1);
    if (!h.startsWith("[")) h = h.split(":", 1)[0];
    return h.replace(WWW, "").replace(/\.+$/, "");
  }

  function encode(s, L) {
    const ids = new Int32Array(L);
    const n = Math.min(s.length, L);
    for (let i = 0; i < n; i++) {
      const c = s.charCodeAt(i);
      ids[i] = c >= 32 && c <= 126 ? c - 30 : 1;
    }
    return ids;
  }

  function f32(b64) {
    const bin = atob(b64);
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }

  function load(M) {
    const W = {};
    for (const [k, v] of Object.entries(M.weights)) W[k] = f32(v);
    const D = M.emb_dim, C = M.channels, ks = M.kernels, H = M.hidden, L = M.max_len;
    const F = C * ks.length;

    function logit(ids) {
      const x = new Float32Array(L * D);                     // [L, D]
      for (let t = 0; t < L; t++) x.set(W["emb.weight"].subarray(ids[t] * D, ids[t] * D + D), t * D);
      const h = new Float32Array(F);
      ks.forEach((k, j) => {
        const w = W[`convs.${j}.weight`], b = W[`convs.${j}.bias`];   // w: [C, D, k]
        for (let c = 0; c < C; c++) {
          let best = 0;                                      // max(relu(.)) >= 0
          for (let t = 0; t + k <= L; t++) {
            let s = b[c];
            for (let d = 0; d < D; d++) {
              const wo = (c * D + d) * k;
              for (let i = 0; i < k; i++) s += w[wo + i] * x[(t + i) * D + d];
            }
            if (s > best) best = s;
          }
          h[j * C + c] = best;
        }
      });
      const w1 = W["fc1.weight"], b1 = W["fc1.bias"], w2 = W["fc2.weight"];
      let out = W["fc2.bias"][0];
      for (let o = 0; o < H; o++) {
        let s = b1[o];
        for (let i = 0; i < F; i++) s += w1[o * F + i] * h[i];
        if (s > 0) out += w2[o] * s;
      }
      return out;
    }

    function score(url) {
      const n = normUrl(url);
      const text = M.input === "host" ? hostOf(n) : n;
      const p = 1 / (1 + Math.exp(-logit(encode(text, L))));
      const t = M.thresholds;
      const verdict = p >= t.block ? "phishing" : p >= t.warn ? "suspicious" : "safe";
      return { host: hostOf(n), input: text, p, verdict };
    }

    return { score, logit, meta: M };
  }

  g.PhishModel = { load, normUrl, hostOf, encode };
})(typeof self !== "undefined" ? self : globalThis);
