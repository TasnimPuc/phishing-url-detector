// Scores every top-level navigation locally and warns on phishing.
importScripts("weights.js", "model.js");
const model = PhishModel.load(self.PHISH_MODEL);
const BADGE = { phishing: ["!", "#d93025"], suspicious: ["?", "#f29900"], safe: ["", "#188038"] };

async function getSettings() {
  const { settings = {} } = await chrome.storage.local.get("settings");
  return { interstitial: true, ...settings };
}

async function isTrusted(host) {
  const { trusted = [] } = await chrome.storage.local.get("trusted");
  const { once = [] } = await chrome.storage.session.get("once");
  return trusted.includes(host) || once.includes(host);
}

async function check(tabId, url) {
  if (!/^https?:\/\//i.test(url)) return;
  const r = model.score(url);
  const [text, color] = BADGE[r.verdict];
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
  chrome.action.setTitle({ tabId, title: `PhishGuard: ${r.verdict} (${(100 * r.p).toFixed(1)}%) - ${r.host}` });
  if (r.verdict === "phishing" && !(await isTrusted(r.host)) && (await getSettings()).interstitial) {
    const page = chrome.runtime.getURL("warning.html");
    chrome.tabs.update(tabId, { url: `${page}?u=${encodeURIComponent(url)}&p=${r.p.toFixed(4)}` });
  }
}

chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  const url = info.url || (info.status === "loading" ? tab.url : null);
  if (url) check(tabId, url);
});
