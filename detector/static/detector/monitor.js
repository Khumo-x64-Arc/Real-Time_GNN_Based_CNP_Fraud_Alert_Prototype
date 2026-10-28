(function () {
  const ctl = document.getElementById("controls");
  const tape = document.getElementById("tape");
  const feed = document.getElementById("feed");
  const playBtn = document.getElementById("play");
  const stateEl = document.getElementById("stream-state");
  const startEl = document.getElementById("start");
  const rateEl = document.getElementById("rate");
  const onlyFlagged = document.getElementById("only-flagged");
  const thr = JSON.parse(document.getElementById("thresholds").textContent);
  const csrf = ctl.dataset.csrf;
  const MAX_ROWS = 300;

  let es = null, pos = 0, total = 0;
  const c = { seen: 0, fraud: 0, review: 0, caught: 0, known: 0 };

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  const money = (v) => (v == null ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const shortTime = (t) => (t.length > 10 ? t.slice(-8) : t);

  function counters() {
    document.getElementById("c-seen").textContent = c.seen.toLocaleString();
    document.getElementById("c-fraud").textContent = c.fraud.toLocaleString();
    document.getElementById("c-review").textContent = c.review.toLocaleString();
    document.getElementById("c-caught").textContent = `${c.caught} / ${c.known}`;
  }

  function addTx(tx) {
    if (onlyFlagged.checked && tx.level === "clear") return;
    const p = tx.probs.GNN;
    const el = document.createElement("div");
    el.className = `tx ${tx.level} new`;
    el.innerHTML = `
      <span class="when">${esc(shortTime(tx.time))}</span>
      <span class="who">${esc(tx.merchant || tx.id)} <small>${esc(tx.category)}${tx.country ? " · " + esc(tx.country) : ""}</small><br><small>${esc(tx.account)}${tx.channel ? " · " + esc(tx.channel) : ""}</small></span>
      <span class="amt">${money(tx.amount)}</span>
      <span class="risk" title="GNN ${pct(p)} · threshold ${pct(thr.GNN)}">
        <span class="meter"><i style="width:${Math.max(2, Math.min(100, p * 100))}%"></i><b style="position:absolute;top:0;bottom:0;width:2px;background:var(--ink);left:${Math.min(100, thr.GNN * 100)}%"></b></span>
        <small>GNN ${pct(p)}${tx.actual === 1 ? " · fraud in data" : ""}</small>
      </span>`;
    const empty = tape.querySelector(".empty");
    if (empty) empty.remove();
    tape.prepend(el);
    while (tape.children.length > MAX_ROWS) tape.lastChild.remove();
  }

  function addAlert(tx) {
    const el = document.createElement("article");
    el.className = `alert-card ${tx.level}`;
    const truth = tx.actual == null ? "" : tx.actual === 1
      ? `<span class="truth hit">Labelled fraud in the dataset</span>` : `<span class="truth miss">Labelled legitimate in the dataset</span>`;
    const reasons = tx.signals.length ? tx.signals.map((s) => `<li>${esc(s.text)}</li>`).join("") : "<li>No behavioural signal; flagged on the GNN score</li>";
    const score = `<span>GNN ${pct(tx.probs.GNN, 1)} ${tx.flagged.GNN ? "≥" : "<"} ${pct(thr.GNN, 1)} threshold</span>`;
    el.innerHTML = `
      <header><span class="lvl">${tx.level === "fraud" ? "Likely fraud" : "Needs review"}</span><b>${money(tx.amount)}</b></header>
      <div class="small">${esc(tx.merchant)} ${tx.country ? "· " + esc(tx.country) : ""}</div>
      <div class="small muted">${esc(tx.id)} · ${esc(tx.account)}</div>
      ${tx.drivers && tx.drivers.length ? `<div class="drivers-mini"><span class="small muted">Top drivers</span>${tx.drivers.map((d) =>
        `<span class="chip ${d.lo > 0 ? "up" : "down"}" title="${esc(d.label)}: ${esc(d.value)} (${d.lo > 0 ? "+" : "−"}${Math.abs(d.lo).toFixed(2)} log-odds)">${esc(d.label)} <b>${d.lo > 0 ? "+" : "−"}${Math.abs(d.lo).toFixed(1)}</b></span>`).join("")}</div>` : ""}
      <ul>${reasons}</ul>
      <div class="scores">${score}</div>
      ${tx.alert_id ? `<a class="small" href="/alerts/${tx.alert_id}/" target="_blank" rel="noopener">Why was this flagged?</a>` : ""}
      <div style="display:flex;justify-content:space-between;align-items:center;gap:.5rem;margin-top:.6rem;flex-wrap:wrap">${truth}
        <span class="review-btns" data-id="${tx.alert_id}">
          <button class="btn small" data-review="confirmed">Confirm</button>
          <button class="btn ghost small" data-review="dismissed">False alarm</button>
        </span></div>`;
    const empty = feed.querySelector(".empty");
    if (empty) empty.remove();
    feed.prepend(el);
    while (feed.children.length > 60) feed.lastChild.remove();
    if (document.hidden && "Notification" in window && Notification.permission === "granted") {
      new Notification(`${tx.level === "fraud" ? "Likely fraud" : "Needs review"}: ${money(tx.amount)}`, { body: tx.signals[0]?.text || tx.id, tag: "fw-" + tx.alert_id });
    }
  }

  feed.addEventListener("click", async (e) => {
    const b = e.target.closest("button[data-review]");
    if (!b) return;
    const wrap = b.parentElement;
    const body = new URLSearchParams({ review: b.dataset.review });
    const r = await fetch(ctl.dataset.reviewUrl.replace("/0/", `/${wrap.dataset.id}/`), {
      method: "POST", body, headers: { "X-CSRFToken": csrf, "x-requested-with": "fetch" },
    });
    if (r.ok) wrap.innerHTML = `<span class="small">${esc((await r.json()).label)}</span>`;
  });

  function stop(msg) {
    if (es) es.close();
    es = null;
    playBtn.textContent = pos && pos < total ? "Resume" : "Start stream";
    if (msg) stateEl.textContent = msg;
  }

  function start() {
    const from = parseInt(startEl.value, 10) || 0;
    pos = from;
    const url = `${ctl.dataset.streamUrl}?rate=${rateEl.value}&start=${pos}`;
    es = new EventSource(url);
    playBtn.textContent = "Pause";
    stateEl.textContent = pos ? `Rebuilding account history up to #${pos}…` : "Connecting…";
    es.addEventListener("meta", (e) => { total = JSON.parse(e.data).total; stateEl.textContent = "Streaming"; });
    es.addEventListener("tx", (e) => {
      const d = JSON.parse(e.data);
      pos = d.pos;
      startEl.value = pos;
      d.rows.forEach((tx) => {
        c.seen++;
        if (tx.actual === 1) { c.known++; if (tx.level === "fraud") c.caught++; }
        if (tx.level === "fraud") c.fraud++;
        if (tx.level === "review") c.review++;
        addTx(tx);
        if (tx.level !== "clear") addAlert(tx);
      });
      stateEl.textContent = `Streaming · ${pos.toLocaleString()} of ${total.toLocaleString()}`;
      counters();
    });
    es.addEventListener("done", () => stop("Reached the end of the test period."));
    // EventSource would silently reconnect from the old start position; stop instead so nothing repeats.
    es.onerror = () => { if (es) stop(`Connection lost at #${pos.toLocaleString()}. Press Resume to continue.`); };
  }

  playBtn.addEventListener("click", () => (es ? stop(`Paused at #${pos.toLocaleString()}`) : start()));
  rateEl.addEventListener("change", () => { if (es) { stop(); start(); } });
  window.addEventListener("beforeunload", () => es && es.close());
})();
