/* Results-page charts: plain SVG, no library, so the app stays dependency-free.
   Data comes from the <script id="chart-data" type="application/json"> block the view renders.
   Every chart re-renders on resize, has a hover tooltip, a legend that toggles models,
   and a data table underneath for screen readers and exact values. */
(() => {
  const dataEl = document.getElementById("chart-data");
  if (!dataEl) return;
  const D = JSON.parse(dataEl.textContent);
  const NS = "http://www.w3.org/2000/svg";
  const hidden = new Set(); // model names switched off in any legend; shared by all charts

  // ------------------------------------------------------------------ helpers
  const el = (tag, attrs = {}, parent, ns = NS) => {
    const n = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "text") n.textContent = v;
      else if (k === "style") n.style.cssText = v;
      else n.setAttribute(k, v);
    }
    if (parent) parent.appendChild(n);
    return n;
  };
  const html = (tag, attrs, parent) => el(tag, attrs, parent, null);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const f3 = (v) => (v == null ? "–" : v.toFixed(3));
  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const int = (v) => Math.round(v).toLocaleString();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // one tooltip for the whole page
  const tip = html("div", { class: "chart-tip", role: "status", "aria-live": "polite", hidden: true }, document.body);
  function showTip(content, evt) {
    tip.innerHTML = content;
    tip.hidden = false;
    const r = tip.getBoundingClientRect(), pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  const swatch = (color, dash) =>
    `<svg class="sw" width="18" height="10" aria-hidden="true"><line x1="1" x2="17" y1="5" y2="5" style="stroke:${color};stroke-dasharray:${dash || "none"}"/></svg>`;

  const redrawers = [];
  const redrawAll = () => redrawers.forEach((f) => f());

  function mount(host, draw) {
    let lastW = 0;
    const run = () => {
      const w = host.clientWidth;
      if (!w) return;
      lastW = w;
      host.replaceChildren();
      draw(w);
    };
    new ResizeObserver(() => { if (Math.abs(host.clientWidth - lastW) > 2) run(); }).observe(host);
    redrawers.push(run);
    run();
  }

  function legend(parent, series, { toggle = true } = {}) {
    const ul = html("ul", { class: "legend" }, parent);
    for (const s of series) {
      const li = html("li", {}, ul);
      const body = `${swatch(s.color, s.dash)}<span>${esc(s.label || s.name)}</span>`;
      if (!toggle) { li.innerHTML = body; continue; }
      const b = html("button", { type: "button", "data-model": s.name, "aria-pressed": String(!hidden.has(s.name)) }, li);
      b.innerHTML = body;
      b.addEventListener("click", () => {
        hidden.has(s.name) ? hidden.delete(s.name) : hidden.add(s.name);
        // keep every legend for the same model in step, then redraw all charts
        document.querySelectorAll(".chart-legend button").forEach((x) => {
          if (x.dataset.model === s.name) x.setAttribute("aria-pressed", String(!hidden.has(s.name)));
        });
        redrawAll();
      });
    }
    ul.classList.add("chart-legend");
    return ul;
  }

  function dataTable(parent, caption, head, rows) {
    const d = html("details", { class: "chart-table" }, parent);
    html("summary", { text: "Data table" }, d);
    const wrap = html("div", { class: "table-wrap" }, d);
    const t = html("table", {}, wrap);
    html("caption", { class: "sr-only", text: caption }, t);
    const tr = html("tr", {}, html("thead", {}, t));
    head.forEach((h, i) => html("th", { class: i ? "n" : null, text: h }, tr));
    const tb = html("tbody", {}, t);
    for (const r of rows) {
      const row = html("tr", {}, tb);
      r.forEach((c, i) => html("td", { class: i ? "n" : null, text: c }, row));
    }
  }

  function interp(points, x) {
    if (!points.length) return null;
    if (x <= points[0][0]) return points[0][1];
    for (let i = 1; i < points.length; i++) {
      const [x1, y1] = points[i];
      if (x1 >= x) {
        const [x0, y0] = points[i - 1];
        return x1 === x0 ? Math.max(y0, y1) : y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
      }
    }
    return points[points.length - 1][1];
  }

  // ------------------------------------------------------------------ line chart
  /* opts: {height, x:{log, min, max, ticks, fmt, label}, y:{min, max, ticks, fmt, label},
            series:[{name, color, dash, points, op:[x,y], opLabel}], refs:[{points, label}],
            hoverTitle(x), valueFmt(y), aria} */
  function lineChart(host, opts) {
    mount(host, (W) => {
      const H = opts.height || 300, m = { l: 48, r: 14, t: opts.y.label ? 26 : 12, b: opts.x.label ? 42 : 26 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const { x, y } = opts;
      const lx = (v) => Math.log10(Math.max(v, x.min));
      const sx = x.log
        ? (v) => m.l + ((lx(v) - lx(x.min)) / (lx(x.max) - lx(x.min))) * iw
        : (v) => m.l + ((v - x.min) / (x.max - x.min)) * iw;
      const ix = x.log
        ? (px) => 10 ** (lx(x.min) + ((px - m.l) / iw) * (lx(x.max) - lx(x.min)))
        : (px) => x.min + ((px - m.l) / iw) * (x.max - x.min);
      const sy = (v) => m.t + (1 - (v - y.min) / (y.max - y.min)) * ih;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": opts.aria }, host);

      for (const t of y.ticks) {
        el("line", { class: "grid", x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }, svg);
        el("text", { x: m.l - 7, y: sy(t), dy: 4, "text-anchor": "end", text: (y.fmt || f3)(t) }, svg);
      }
      for (const t of x.ticks) {
        el("line", { class: "grid", x1: sx(t), x2: sx(t), y1: m.t, y2: m.t + ih }, svg);
        el("text", { x: sx(t), y: m.t + ih + 16, "text-anchor": "middle", text: (x.fmt || String)(t) }, svg);
      }
      el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }, svg);
      if (x.label) el("text", { x: m.l + iw, y: H - 4, "text-anchor": "end", text: x.label }, svg);
      if (y.label) el("text", { x: m.l - 7, y: 10, "text-anchor": "start", class: "ylabel", text: `↑ ${y.label}` }, svg);

      const clip = `clip-${Math.random().toString(36).slice(2)}`;
      el("rect", { x: m.l, y: m.t - 3, width: iw + 3, height: ih + 6 }, el("clipPath", { id: clip }, el("defs", {}, svg)));
      const plot = el("g", { "clip-path": `url(#${clip})` }, svg);
      const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(clamp(p[1], y.min, y.max)).toFixed(1)}`).join("");
      for (const r of opts.refs || []) {
        el("path", { class: "ref", d: path(r.points) }, plot);
        if (r.label) el("text", { class: "ref-label", x: sx(r.at[0]), y: sy(r.at[1]), dy: -5, "text-anchor": r.anchor || "start", text: r.label }, plot);
      }
      const visible = opts.series.filter((s) => !hidden.has(s.name) && s.points && s.points.length);
      for (const s of visible) {
        el("path", { class: "curve", d: path(s.points), style: `stroke:${s.color};stroke-dasharray:${s.dash || "none"}` }, plot);
      }
      for (const s of visible) {
        if (!s.op) continue;
        el("circle", { class: "op", cx: sx(clamp(s.op[0], x.min, x.max)), cy: sy(s.op[1]), r: 4.5, style: `fill:${s.color}` }, svg);
        if (s.opLabel) el("text", { class: "op-label", x: sx(clamp(s.op[0], x.min, x.max)) + 8, y: sy(s.op[1]), dy: s.op[1] > (y.min + y.max) / 2 ? 14 : -6, text: s.opLabel }, svg);
      }

      // hover: crosshair + every visible model's value at that x
      const cross = el("line", { class: "crosshair", y1: m.t, y2: m.t + ih, visibility: "hidden" }, svg);
      const dots = visible.map((s) => el("circle", { class: "hover-dot", r: 4, visibility: "hidden", style: `fill:${s.color}` }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (e) => {
        const box = svg.getBoundingClientRect();
        const px = clamp(((e.clientX - box.left) / box.width) * W, m.l, m.l + iw);
        const xv = ix(px);
        cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
        const vals = visible.map((s, i) => {
          const v = interp(s.points, xv);
          dots[i].setAttribute("cx", px); dots[i].setAttribute("cy", sy(clamp(v, y.min, y.max)));
          dots[i].setAttribute("visibility", "visible");
          return { s, v };
        }).sort((a, b) => b.v - a.v);
        showTip(`<b>${esc(opts.hoverTitle(xv))}</b>` + vals.map(({ s, v }) =>
          `<div class="row">${swatch(s.color, s.dash)}<span>${esc(s.name)}</span><b>${(opts.valueFmt || f3)(v)}</b></div>`).join(""), e);
      });
      hit.addEventListener("pointerleave", () => {
        cross.setAttribute("visibility", "hidden");
        dots.forEach((d) => d.setAttribute("visibility", "hidden"));
        hideTip();
      });
    });
  }

  // ------------------------------------------------------------------ horizontal bars
  /* rows: [{label, color, segs:[{v, cls, tip}], end}]  — `end` is the direct label after the bar */
  function hBars(host, { rows, max, ticks, fmt, aria, axisLabel, rowH = 30, barH = 14, group }) {
    mount(host, (W) => {
      const labelW = Math.min(110, W * 0.28), endW = 64;
      const m = { l: labelW, r: endW, t: 6, b: 24 };
      const iw = W - m.l - m.r;
      let yCur = m.t;
      const layout = rows.map((r) => {
        if (r.header) { if (yCur > m.t) yCur += 10; const o = { r, y: yCur + 12 }; yCur += 22; return o; }
        const o = { r, y: yCur }; yCur += rowH; return o;
      });
      const H = yCur + m.b;
      const sx = (v) => (v / max) * iw;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": aria }, host);
      for (const t of ticks) {
        el("line", { class: "grid", x1: m.l + sx(t), x2: m.l + sx(t), y1: m.t, y2: H - m.b }, svg);
        el("text", { x: m.l + sx(t), y: H - m.b + 15, "text-anchor": "middle", text: fmt(t) }, svg);
      }
      if (axisLabel) el("text", { x: W - 2, y: H - m.b + 15, "text-anchor": "end", text: axisLabel }, svg);
      el("line", { class: "axis", x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }, svg);
      for (const { r, y } of layout) {
        if (r.header) { el("text", { class: "group-label", x: 0, y, text: r.header }, svg); continue; }
        const off = hidden.has(r.name);
        const cy = y + (rowH - barH) / 2;
        el("text", { class: "row-label", x: labelW - 10, y: cy + barH / 2, dy: 4, "text-anchor": "end", text: r.label, opacity: off ? 0.4 : 1 }, svg);
        if (off) continue;
        let x0 = m.l;
        r.segs.forEach((s, i) => {
          const w = Math.max(0, sx(s.v) - (i < r.segs.length - 1 && s.v > 0 ? 2 : 0)); // 2px gap between segments
          const last = i === r.segs.length - 1 || r.segs.slice(i + 1).every((q) => !q.v);
          const rad = last ? Math.min(4, w) : 0;
          const d = `M${x0},${cy}h${w - rad}${rad ? `a${rad},${rad} 0 0 1 ${rad},${rad}v${barH - 2 * rad}a${rad},${rad} 0 0 1 ${-rad},${rad}` : `v${barH}`}h${-(w - rad)}z`;
          if (w > 0) {
            const p = el("path", { d, class: `seg ${s.cls || ""}`, style: s.cls ? null : `fill:${r.color}` }, svg);
            p.addEventListener("pointermove", (e) => showTip(s.tip, e));
            p.addEventListener("pointerleave", hideTip);
          }
          x0 += sx(s.v);
        });
        if (r.end) el("text", { class: "end-label", x: x0 + 6, y: cy + barH / 2, dy: 4, text: r.end }, svg);
      }
    });
  }

  // ------------------------------------------------------------------ charts
  const models = D.models;
  const base = models.filter((s) => s.name !== "Combined");
  const sel = (id) => document.querySelector(`[data-chart="${id}"] .chart`);
  const panel = (id) => document.querySelector(`[data-chart="${id}"]`);
  const legendSeries = models.map((s) => ({ name: s.name, color: s.color, dash: s.dash }));

  // ROC (log FPR: fraud models only separate below 1% false-positive rate)
  if (sel("roc")) {
    lineChart(sel("roc"), {
      aria: "ROC curves for the GNN on a log false-positive-rate axis",
      x: { log: true, min: 1e-4, max: 1, ticks: [1e-4, 1e-3, 1e-2, 1e-1, 1], fmt: (v) => (v >= 0.01 ? `${v * 100}%` : `${+(v * 100).toPrecision(1)}%`), label: "False-positive rate (log scale)" },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => `${v * 100}%`, label: "Frauds caught" },
      series: models.map((s) => ({ ...s, points: s.roc, op: s.op_roc })),
      hoverTitle: (x) => `At ${pct(x, x < 0.01 ? 2 : 1)} false positives`, valueFmt: (v) => `${pct(v)} caught`,
    });
    legend(panel("roc"), models.map((s) => ({ ...s, label: `${s.name} · ${f3(s.auc)}` })));
    dataTable(panel("roc"), "ROC-AUC by model", ["Model", "ROC-AUC", "Recall at threshold", "False-positive rate at threshold"],
      models.map((s) => [s.name, f3(s.auc), pct(s.op_roc[1]), pct(s.op_roc[0], 3)]));
  }

  // Precision–recall
  if (sel("pr")) {
    const withPr = models.filter((s) => s.pr);
    if (!withPr.length) {
      sel("pr").innerHTML = `<p class="empty small">Precision–recall curves are saved during training. Retrain this dataset to see them.</p>`;
    } else {
      lineChart(sel("pr"), {
        aria: "Precision-recall curves for the GNN",
        x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => `${v * 100}%`, label: "Recall (frauds caught)" },
        y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => `${v * 100}%`, label: "Precision" },
        series: withPr.map((s) => ({ ...s, points: s.pr, op: s.op_pr })),
        refs: [{ points: [[0, D.base_rate], [1, D.base_rate]], label: `No skill (${pct(D.base_rate, 2)} fraud)`, at: [0.01, D.base_rate], anchor: "start" }],
        hoverTitle: (x) => `At ${pct(x, 0)} recall`, valueFmt: (v) => `${pct(v)} precision`,
      });
      legend(panel("pr"), withPr.map((s) => ({ ...s, label: `${s.name} · ${f3(s.ap)}` })));
      dataTable(panel("pr"), "Precision and recall at the alert threshold", ["Evaluation", "PR-AUC", "Precision at threshold", "Recall at threshold"],
        withPr.map((s) => [s.name, f3(s.ap), pct(s.op_pr[1]), pct(s.op_pr[0])]));
    }
  }

  // Frauds caught vs missed, and false alarms
  if (sel("caught")) {
    const frauds = models[0].tp + models[0].fn;
    hBars(sel("caught"), {
      aria: "Frauds caught and missed by the GNN",
      max: frauds, ticks: niceTicks(0, frauds, 4).filter((t) => t <= frauds), fmt: int,
      rows: models.map((s) => ({
        name: s.name, label: s.name, color: s.color, end: pct(s.tp / (frauds || 1), 0),
        segs: [
          { v: s.tp, tip: `<b>${esc(s.name)}</b><div>Caught ${int(s.tp)} of ${int(frauds)} frauds</div>` },
          { v: s.fn, cls: "missed", tip: `<b>${esc(s.name)}</b><div>Missed ${int(s.fn)} of ${int(frauds)} frauds</div>` },
        ],
      })),
    });
    legend(panel("caught"), [{ name: "caught", color: "var(--ink)", label: "Caught (model colour)" }, { name: "missed", color: "var(--line)", label: "Missed" }], { toggle: false });
    const maxFp = Math.max(1, ...models.map((s) => s.fp));
    hBars(document.querySelector(`[data-chart-part="alarms"]`), {
      aria: "False alarms raised by the GNN",
      max: niceTicks(0, maxFp, 4).slice(-1)[0] >= maxFp ? niceTicks(0, maxFp, 4).slice(-1)[0] : maxFp,
      ticks: niceTicks(0, maxFp, 4), fmt: int,
      rows: models.map((s) => ({
        name: s.name, label: s.name, color: s.color, end: int(s.fp),
        segs: [{ v: s.fp, tip: `<b>${esc(s.name)}</b><div>${int(s.fp)} false alarms among ${int(s.fp + s.tn)} legitimate transactions (${pct(s.fp / (s.fp + s.tn || 1), 2)})</div>` }],
      })),
    });
    dataTable(panel("caught"), "Confusion-matrix counts at the alert threshold", ["Evaluation", "Caught (TP)", "Missed (FN)", "False alarms (FP)", "Correctly passed (TN)", "Threshold"],
      models.map((s) => [s.name, int(s.tp), int(s.fn), int(s.fp), int(s.tn), f3(s.thr)]));
  }

  // Score distributions: one small chart per model
  if (panel("hist")) {
    const grid = panel("hist").querySelector(".multiples");
    const withH = models.filter((s) => s.hist_fraud);
    if (!withH.length) {
      grid.innerHTML = `<p class="empty small">Score distributions are saved during training. Retrain this dataset to see them.</p>`;
    } else {
      const bins = D.bins;
      for (const s of withH) {
        const card = html("figure", { class: "multiple" }, grid);
        html("figcaption", {}, card).innerHTML = `${swatch(s.color, s.dash)} ${esc(s.name)}`;
        const host = html("div", { class: "chart" }, card);
        const nf = s.hist_fraud.reduce((a, b) => a + b, 0) || 1, nl = s.hist_legit.reduce((a, b) => a + b, 0) || 1;
        const sf = s.hist_fraud.map((c) => c / nf), sl = s.hist_legit.map((c) => c / nl);
        mount(host, (W) => {
          const H = 170, m = { l: 36, r: 8, t: 8, b: 24 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
          const sx = (v) => m.l + v * iw, sy = (v) => m.t + (1 - v) * ih, bw = iw / bins;
          const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
            "aria-label": `${s.name}: share of frauds and of legitimate transactions in each score band` }, host);
          for (const t of [0, 0.5, 1]) {
            el("line", { class: "grid", x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }, svg);
            el("text", { x: m.l - 5, y: sy(t), dy: 4, "text-anchor": "end", text: `${t * 100}%` }, svg);
          }
          for (const t of [0, 0.5, 1]) el("text", { x: sx(t), y: H - 6, "text-anchor": t ? (t === 1 ? "end" : "middle") : "start", text: t.toFixed(1) }, svg);
          sf.forEach((v, i) => {
            if (v <= 0) return;
            const h = Math.max(1, v * ih), r = Math.min(3, h);
            const x0 = sx(i / bins) + 1, w = bw - 2;
            el("path", { class: "hist-fraud", d: `M${x0},${sy(0)}v${-(h - r)}a${r},${r} 0 0 1 ${r},${-r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z` }, svg);
          });
          el("path", { class: "hist-legit", d: `M${sx(0)},${sy(0)}` + sl.map((v, i) => `L${sx(i / bins)},${sy(v)}H${sx((i + 1) / bins)}`).join("") + `V${sy(0)}Z` }, svg);
          el("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: sy(0), y2: sy(0) }, svg);
          el("line", { class: "thr", x1: sx(s.thr), x2: sx(s.thr), y1: m.t, y2: sy(0) }, svg);
          el("text", { class: "thr-label", x: sx(s.thr) + (s.thr > 0.75 ? -4 : 4), y: m.t + 10, "text-anchor": s.thr > 0.75 ? "end" : "start", text: `alert ≥ ${s.thr.toFixed(2)}` }, svg);
          const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
          hit.addEventListener("pointermove", (e) => {
            const box = svg.getBoundingClientRect();
            const i = clamp(Math.floor(((((e.clientX - box.left) / box.width) * W - m.l) / iw) * bins), 0, bins - 1);
            showTip(`<b>${esc(s.name)} score ${(i / bins).toFixed(2)}–${((i + 1) / bins).toFixed(2)}</b>` +
              `<div class="row"><span class="sq fraud"></span><span>Frauds</span><b>${pct(sf[i])} (${int(s.hist_fraud[i])})</b></div>` +
              `<div class="row"><span class="sq legit"></span><span>Legitimate</span><b>${pct(sl[i], 2)} (${int(s.hist_legit[i])})</b></div>`, e);
          });
          hit.addEventListener("pointerleave", hideTip);
        });
      }
      dataTable(panel("hist"), "Share of frauds and legitimate transactions per score band",
        ["Score band", ...withH.flatMap((s) => [`${s.name} frauds`, `${s.name} legit`])],
        Array.from({ length: bins }, (_, i) => [`${(i / bins).toFixed(2)}–${((i + 1) / bins).toFixed(2)}`,
          ...withH.flatMap((s) => [int(s.hist_fraud[i]), int(s.hist_legit[i])])]));
    }
  }

  // Training curves: validation PR-AUC as the GNN learned
  if (panel("training")) {
    const grid = panel("training").querySelector(".multiples");
    const tr = D.training || {};
    const names = Object.keys(tr).filter((n) => tr[n].points && tr[n].points.length > 1);
    if (!names.length) {
      grid.innerHTML = `<p class="empty small">Training curves are saved during training. Retrain this dataset to see them.</p>`;
    } else {
      for (const n of names) {
        const s = models.find((x) => x.name === n) || { color: "var(--ink)" };
        const pts = tr[n].points, unit = tr[n].unit;
        const best = pts.reduce((a, b) => (b[1] > a[1] ? b : a));
        const ys = pts.map((p) => p[1]);
        const lo = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10), hi = Math.min(1, Math.ceil(Math.max(...ys) * 10) / 10 || 1);
        const card = html("figure", { class: "multiple" }, grid);
        html("figcaption", {}, card).innerHTML = `${swatch(s.color)} ${esc(n)} <span class="muted">· best ${f3(best[1])} at ${unit} ${best[0]}</span>`;
        const host = html("div", { class: "chart" }, card);
        const xmax = pts[pts.length - 1][0], xmin = pts[0][0];
        lineChart(host, {
          height: 170, aria: `${n} validation PR-AUC by ${unit}`,
          x: { min: xmin, max: xmax, ticks: niceTicks(xmin, xmax, 4).filter((t) => t >= xmin && t <= xmax), fmt: int, label: `${unit}s` },
          y: { min: lo, max: hi === lo ? lo + 0.1 : hi, ticks: niceTicks(lo, hi === lo ? lo + 0.1 : hi, 3), fmt: (v) => v.toFixed(2) },
          series: [{ name: n, color: s.color, points: pts, op: best }],
          hoverTitle: (x) => `${unit[0].toUpperCase()}${unit.slice(1)} ${Math.round(x)}`, valueFmt: (v) => `PR-AUC ${f3(v)}`,
        });
      }
      dataTable(panel("training"), "Validation PR-AUC during training", ["Model", "Unit", "Steps recorded", "Best PR-AUC", "At"],
        names.map((n) => { const p = tr[n].points, b = p.reduce((a, c) => (c[1] > a[1] ? c : a)); return [n, tr[n].unit, String(p.length), f3(b[1]), String(b[0])]; }));
    }
  }

  // F1 across datasets
  if (sel("cross") && D.cross.length) {
    const names = D.model_order.filter((n) => D.cross.some((c) => n in c.values));
    const color = Object.fromEntries(D.model_order.map((n) => [n, D.colors[n]]));
    const rows = [];
    for (const c of D.cross) {
      rows.push({ header: c.dataset });
      for (const n of names) {
        if (!(n in c.values)) continue;
        rows.push({ name: n, label: n, color: color[n], end: f3(c.values[n]),
          segs: [{ v: c.values[n], tip: `<b>${esc(c.dataset)}</b><div>${esc(n)}: F1 ${f3(c.values[n])}</div>` }] });
      }
    }
    hBars(sel("cross"), { aria: "F1 by model for every trained dataset", rows, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => v.toFixed(2), rowH: 20, barH: 11, axisLabel: "" });
    legend(panel("cross"), names.map((n) => ({ name: n, color: color[n], dash: D.dashes[n] })));
  }
})();
