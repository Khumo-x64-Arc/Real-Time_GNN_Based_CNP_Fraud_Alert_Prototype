// Polls for new alerts on every page: toast + optional desktop notification + badge count.
(function () {
  const root = document.body;
  const url = root.dataset.alertsUrl;
  const listUrl = root.dataset.alertListUrl;
  const toasts = document.getElementById("toasts");
  const badge = document.getElementById("alert-badge");
  const notifyBtn = document.getElementById("notify-btn");
  let lastId = 0;

  const money = (v) => (v == null ? "" : Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  window.fwToast = function (a) {
    const el = document.createElement("div");
    el.className = "toast " + a.level;
    el.setAttribute("role", "alert");
    el.innerHTML = `<strong>${esc(a.level_label)} · ${esc(money(a.amount))}</strong>
      <div class="small">${esc(a.reason)}</div>
      <div class="small muted">${esc(a.tx)}${a.merchant ? " at " + esc(a.merchant) : ""}</div>
      <a href="${listUrl}">Review alerts</a>`;
    toasts.prepend(el);
    while (toasts.children.length > 4) toasts.lastChild.remove();
    setTimeout(() => el.remove(), 9000);
  };

  function desktop(a) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    const n = new Notification(`${a.level_label}: ${money(a.amount)}`, { body: `${a.reason}\n${a.tx}`, tag: "fw-" + a.id });
    n.onclick = () => { window.focus(); location.href = listUrl; };
  }

  function setBadge(n) {
    if (!badge) return;
    badge.hidden = !n;
    badge.textContent = n > 99 ? "99+" : n;
  }

  async function poll() {
    try {
      const r = await fetch(`${url}?after=${lastId}`, { headers: { Accept: "application/json" } });
      const d = await r.json();
      setBadge(d.open);
      // The live monitor page shows its own alerts inline; only toast when elsewhere or tab hidden.
      const onMonitor = !!document.getElementById("tape");
      if (lastId) {
        d.alerts.slice().reverse().forEach((a) => {
          if (!onMonitor) window.fwToast(a);
          if (document.hidden || !onMonitor) desktop(a);
        });
      }
      lastId = d.last_id;
    } catch (e) { /* server restarting; try again next tick */ }
  }

  if (notifyBtn && "Notification" in window && Notification.permission === "default") {
    notifyBtn.hidden = false;
    notifyBtn.addEventListener("click", async () => {
      await Notification.requestPermission();
      notifyBtn.hidden = true;
    });
  }

  poll();
  setInterval(poll, 4000);
})();
