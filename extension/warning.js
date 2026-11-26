const q = new URLSearchParams(location.search);
const url = q.get("u") || "";
const host = PhishModel.hostOf(PhishModel.normUrl(url));
document.getElementById("host").textContent = host;
document.getElementById("p").textContent = (100 * Number(q.get("p") || 0)).toFixed(1) + "%";
document.getElementById("back").onclick = () => (history.length > 2 ? history.go(-2) : (location.href = "about:blank"));
document.getElementById("go").onclick = async () => {
  const { once = [] } = await chrome.storage.session.get("once");
  if (!once.includes(host)) once.push(host);
  await chrome.storage.session.set({ once });          // allowed for this browser session only
  location.href = url;
};
