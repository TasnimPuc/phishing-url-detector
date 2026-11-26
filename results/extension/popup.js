const model = PhishModel.load(self.PHISH_MODEL);
const COLORS = { safe: "#188038", suspicious: "#f29900", phishing: "#d93025" };
const $ = (id) => document.getElementById(id);

(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab && tab.url ? tab.url : "";
  if (/^https?:\/\//i.test(url)) {
    const r = model.score(url);
    $("host").textContent = r.host;
    $("verdict").textContent = r.verdict;
    $("verdict").className = "verdict " + r.verdict;
    $("fill").style.width = (100 * r.p).toFixed(1) + "%";
    $("fill").style.background = COLORS[r.verdict];
    $("prob").textContent = `Phishing probability: ${(100 * r.p).toFixed(1)}%`;
    $("trust").onclick = async () => {
      const { trusted = [] } = await chrome.storage.local.get("trusted");
      if (!trusted.includes(r.host)) trusted.push(r.host);
      await chrome.storage.local.set({ trusted });
      $("trust").textContent = "Trusted";
    };
  } else {
    $("host").textContent = "Not a web page";
    $("trust").disabled = true;
  }
  const { settings = {} } = await chrome.storage.local.get("settings");
  $("inter").checked = settings.interstitial !== false;
  $("inter").onchange = () =>
    chrome.storage.local.set({ settings: { ...settings, interstitial: $("inter").checked } });
  const m = model.meta;
  $("meta").textContent = `Model: char-CNN on ${m.input} | test AUROC ${m.metrics.auroc.toFixed(3)} | runs offline`;
})();
