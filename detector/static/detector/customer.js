// Customer dashboard: live feed for one account, alerts, and "was this you?" replies.
(function () {
  const root = document.getElementById("cust");
  const activity = document.getElementById("activity");
  const attention = document.getElementById("attention");
  const attWrap = document.getElementById("attention-wrap");
  const statusEl = document.getElementById("status");
  const statLine = document.getElementById("stat-line");
  const emptyNote = document.getElementById("empty-note");
  const toasts = document.getElementById("toasts");
  const notifyBtn = document.getElementById("notify-btn");
  const csrf = root.dataset.csrf;
  let lastId = parseInt(root.dataset.lastId, 10) || 0;
  const baseTitle = document.title;
  let unseen = 0;

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const money = (v) => (v == null ? "–" : Number(v).toLocaleString("en-ZA", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const nice = (s) => String(s || "").replace(/_/g, " ");
  const when = (t) => { const d = new Date(String(t).replace(" ", "T")); return isNaN(d) ? esc(t) : d.toLocaleString("en-ZA", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); };

  function row(t, fresh) {
    const el = document.createElement("div");
    el.className = `act ${t.status}${fresh ? " new" : ""}`;
    el.innerHTML = `
      <span class="act-icon" aria-hidden="true">${esc((t.merchant || "?").trim().charAt(0))}</span>
      <span class="act-main"><b>${esc(t.merchant || "Unknown merchant")}</b>
        <span class="small muted">${esc(nice(t.category))}${t.country ? " · " + esc(t.country) : ""}${t.channel ? " · " + esc(t.channel) : ""} · ${when(t.time)}</span></span>
      <span class="act-right"><b>${money(t.amount)}</b>
        ${t.status === "history" ? "" : `<span class="pill ${t.status}">${esc(t.status_label)}</span>`}</span>`;
    return el;
  }

  function attnCard(t) {
    const el = document.createElement("article");
    el.className = `attn ${t.status}`;
    const reasons = t.reasons.length ? t.reasons.map((r) => `<li>${esc(r)}</li>`).join("") : "<li>It doesn't match how you normally use your card.</li>";
    el.innerHTML = `
      <header><span class="attn-tag">${t.status === "flagged" ? "Possible fraud" : "Please check"}</span><span class="small muted">${when(t.time)}</span></header>
      <p class="attn-amount">${money(t.amount)}</p>
      <p class="attn-where">${esc(t.merchant || "Unknown merchant")}${t.country ? " · " + esc(t.country) : ""}${t.channel ? " · " + esc(t.channel) : ""}</p>
      <p class="small" style="margin:.8rem 0 .2rem"><b>Why we're asking</b></p>
      <ul class="small">${reasons}</ul>
      <p class="attn-q"><b>Did you make this transaction?</b></p>
      <div class="actions" data-alert="${t.alert_id}">
        <button class="btn" data-r="mine">Yes, this was me</button>
        <button class="btn danger" data-r="not_mine">No, I don't recognise it</button>
      </div>`;
    return el;
  }

  function renderAttention(list) {
    attention.replaceChildren(...list.map(attnCard));
    attWrap.hidden = !list.length;
    const n = list.length;
    statusEl.className = "cust-status " + (n ? "alarm" : "calm");
    statusEl.innerHTML = n
      ? `<b>${n} transaction${n > 1 ? "s need" : " needs"} your attention.</b> Tell us whether you made ${n > 1 ? "them" : "it"} so we can protect your account.`
      : `<b>No suspicious activity.</b> We'll alert you here${"Notification" in window && Notification.permission === "granted" ? ", in your browser" : ""} and by email the moment something looks unusual.`;
  }

  function toast(t) {
    const el = document.createElement("div");
    el.className = "toast " + (t.status === "flagged" ? "" : "review");
    el.setAttribute("role", "alert");
    el.innerHTML = `<strong>${t.status === "flagged" ? "Possible fraud on your card" : "Please check this transaction"}</strong>
      <div class="small">${money(t.amount)} at ${esc(t.merchant || "unknown merchant")}${t.country ? " (" + esc(t.country) + ")" : ""}</div>
      <div class="small muted">${esc(t.reasons[0] || "Unusual for your account")}</div>`;
    toasts.prepend(el);
    setTimeout(() => el.remove(), 10000);
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification(t.status === "flagged" ? "Possible fraud on your card" : "Please check a transaction",
        { body: `${money(t.amount)} at ${t.merchant || "unknown merchant"}. Was this you?`, tag: "cust-" + t.id });
    }
    if (document.hidden) { unseen++; document.title = `(${unseen}) ${baseTitle}`; }
  }

  function addTransactions(list, fresh) {
    list.forEach((t) => {
      activity.prepend(row(t, fresh));
      if (fresh && (t.status === "flagged" || t.status === "review")) toast(t);
    });
    while (activity.children.length > 100) activity.lastChild.remove();
    emptyNote.hidden = activity.children.length > 0;
  }

  attention.addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-r]");
    if (!b) return;
    const box = b.parentElement;
    box.querySelectorAll("button").forEach((x) => (x.disabled = true));
    const r = await fetch(root.dataset.respondUrl.replace("/0/", `/${box.dataset.alert}/`), {
      method: "POST", body: new URLSearchParams({ response: b.dataset.r }),
      headers: { "X-CSRFToken": csrf, "x-requested-with": "fetch" },
    });
    const d = await r.json().catch(() => ({}));
    box.outerHTML = `<p class="flash">${esc(d.message || "Something went wrong. Please try again.")}</p>`;
  });

  async function poll() {
    try {
      const r = await fetch(`${root.dataset.feedUrl}?after=${lastId}`, { headers: { Accept: "application/json" } });
      if (r.redirected || !r.ok) return;
      const d = await r.json();
      if (d.transactions.length) {
        addTransactions(d.transactions, true);
        lastId = d.transactions[d.transactions.length - 1].id;
      }
      // don't wipe a card the customer is answering right now
      if (!attention.querySelector("button:disabled")) renderAttention(d.attention);
      statLine.textContent = `${d.n_monitored} checked, ${d.n_flagged} flagged`;
    } catch (e) { /* offline or server restarting */ }
  }

  const init = JSON.parse(document.getElementById("initial").textContent);
  addTransactions(init.transactions, false);
  renderAttention(init.attention);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { unseen = 0; document.title = baseTitle; } });
  if (notifyBtn && "Notification" in window && Notification.permission === "default") {
    notifyBtn.hidden = false;
    notifyBtn.addEventListener("click", async () => { await Notification.requestPermission(); notifyBtn.hidden = true; poll(); });
  }
  setInterval(poll, 3000);
})();
