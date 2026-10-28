/* The whole transaction graph on a canvas (SVG can't hold hundreds of thousands of nodes).
   Layout comes precomputed from training (detector/ml/txgraph.py); this file only draws and handles
   interaction: pan/zoom (mouse, wheel, touch, keyboard), hover, click-to-trace, search, rings, filters. */
(() => {
  const root = document.getElementById("txgraph");
  if (!root) return;
  const $ = (sel) => root.querySelector(sel);
  const wrap = $(".tg-canvas"), statusEl = $(".tg-status"), panel = $(".tg-details");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const int = (v) => Math.round(v).toLocaleString();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  // labelled datasets say "fraud"; unlabelled transfer runs only know what the GNN flagged
  const FLAGS = root.dataset.mode === "flags";
  const WD = FLAGS
    ? { one: "flagged", many: "flagged", adj: "flagged", yes: "Flagged by the GNN", no: "Not flagged", key: "Flagged transaction",
        entKey: "Entity in a flagged transaction", ring: "Suspected ring", rings: "suspected rings", usedIn: "in flagged transactions",
        verdict: "GNN verdict", label: "flagged", others: "Unflagged transactions" }
    : { one: "fraud", many: "frauds", adj: "fraudulent", yes: "Fraud", no: "Legitimate", key: "Fraud transaction",
        entKey: "Entity used in fraud", ring: "Fraud ring", rings: "fraud rings", usedIn: "in fraud",
        verdict: "Label", label: "fraud", others: "Legitimate transactions" };
  const plural = (s) => (/(s|ss|x|ch|sh)$/i.test(s) ? `${s}es` : `${s}s`);

  let G = null;   // loaded graph
  let started = false;
  const start = () => { if (!started) { started = true; load(); } };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: "400px" });
    io.observe(root);
  } else start();

  async function load() {
    statusEl.textContent = "Loading the transaction graph…";
    try {
      const [metaRes, dataRes] = await Promise.all([fetch(root.dataset.meta), fetch(root.dataset.data)]);
      if (!metaRes.ok || !dataRes.ok) throw new Error(metaRes.ok ? await dataRes.text() : await metaRes.text());
      const meta = await metaRes.json(), buf = await dataRes.arrayBuffer();
      const A = {};
      const ctor = { float32: Float32Array, int32: Int32Array, uint32: Uint32Array, uint8: Uint8Array };
      for (const [k, v] of Object.entries(meta.arrays)) A[k] = new ctor[v.dtype](buf, v.offset, v.length);
      const tp = performance.now();
      G = prepare(meta, A);
      root.dataset.prepareMs = Math.round(performance.now() - tp);
      statusEl.textContent = "";
      setup();
    } catch (e) {
      statusEl.innerHTML = `<span class="tg-error">The graph couldn't be loaded. ${esc(e.message || e).slice(0, 200)}</span>`;
    }
  }

  // ---------------------------------------------------------------- data
  function prepare(meta, A) {
    const N = meta.n_tx, E = meta.n_entities, T = meta.types.length;
    const hub = meta.types.map((t) => meta.hub_types.includes(t));
    // entity -> transactions (CSR)
    const cnt = new Uint32Array(E + 1);
    for (let i = 0; i < A.tx_ent.length; i++) { const e = A.tx_ent[i]; if (e >= 0) cnt[e + 1]++; }
    for (let e = 0; e < E; e++) cnt[e + 1] += cnt[e];
    const fill = cnt.slice(0, E), entTx = new Int32Array(cnt[E]);
    for (let i = 0; i < N; i++) for (let t = 0; t < T; t++) { const e = A.tx_ent[i * T + t]; if (e >= 0) entTx[fill[e]++] = i; }
    // spatial grid over all nodes for hover / click (id < N: transaction, id >= N: entity N + e)
    const [x0, y0, x1, y1] = meta.bounds;
    const span = Math.max(x1 - x0, y1 - y0) || 1, gs = Math.ceil(Math.sqrt((N + E) / 6)) || 1;
    const cell = span / gs, gw = Math.ceil((x1 - x0) / cell) + 1, gh = Math.ceil((y1 - y0) / cell) + 1;
    const cellOf = (x, y) => clamp(Math.floor((y - y0) / cell), 0, gh - 1) * gw + clamp(Math.floor((x - x0) / cell), 0, gw - 1);
    const gStart = new Uint32Array(gw * gh + 1), ids = new Int32Array(N + E), cells = new Uint32Array(N + E);
    for (let i = 0; i < N; i++) { cells[i] = cellOf(A.tx_xy[2 * i], A.tx_xy[2 * i + 1]); gStart[cells[i] + 1]++; }
    for (let e = 0; e < E; e++) { cells[N + e] = cellOf(A.ent_xy[2 * e], A.ent_xy[2 * e + 1]); gStart[cells[N + e] + 1]++; }
    for (let c = 0; c < gw * gh; c++) gStart[c + 1] += gStart[c];
    const gf = gStart.slice(0, gw * gh);
    for (let id = 0; id < N + E; id++) ids[gf[cells[id]]++] = id;
    const lower = meta.names.map((s) => s.toLowerCase());
    return { meta, A, N, E, T, hub, entOff: cnt, entTx, grid: { x0, y0, cell, gw, gh, start: gStart, ids }, lower };
  }

  // ---------------------------------------------------------------- state
  const view = { k: 1, cx: 0, cy: 0 };
  let W = 0, H = 0, dpr = 1;
  let base, overlay, bctx, octx;           // base layer (graph) + overlay (hover ring)
  let snapshot = null, snapView = null;    // bitmap reused while panning / zooming
  let visibleType = [], showLegit = true, selection = null, hoverId = -1, idleTimer = 0;
  let colors = {};

  const typeColorVar = (t, i) => {
    const m = G.meta;
    if (m.hub_types.includes(t)) return "--g-hub";
    return { account: "--g-account", card: "--g-account", device: "--g-device", ip: "--g-ip" }[t]
      || (t === m.anchor_type ? "--g-account" : "--g-hub");
  };
  const SHAPES = { account: "circle", card: "square", device: "diamond", ip: "triangle", merchant: "hollow-square", terminal: "plus" };
  const shapeOf = (t) => SHAPES[t] || (t === G.meta.anchor_type ? "circle" : "hollow-circle");

  function readColors() {
    const cs = getComputedStyle(root);
    const v = (n) => cs.getPropertyValue(n).trim() || "#888";
    colors = { panel: v("--panel"), ink: v("--ink"), muted: v("--muted"), line: v("--line"), fraud: v("--fraud"),
      types: G.meta.types.map((t, i) => v(typeColorVar(t, i))) };
  }

  function setup() {
    const m = G.meta;
    visibleType = m.types.map((t) => !(m.hub_types.includes(t) && G.N > 60000));
    readColors();
    buildControls();
    buildRings();
    $(".tg-stats").innerHTML = `
      <div><b>${int(m.n_tx)}</b><span>transactions</span></div>
      <div><b>${int(m.n_entities)}</b><span>entities</span></div>
      <div><b>${int(m.n_edges)}</b><span>links</span></div>
      <div><b>${int(m.n_fraud)}</b><span>${WD.many}</span></div>
      <div><b>${int(m.clusters)}</b><span>identity clusters</span></div>
      <div><b>${int(m.n_rings)}</b><span>${WD.rings}</span></div>`;
    wrap.replaceChildren();
    base = document.createElement("canvas");
    overlay = document.createElement("canvas");
    base.className = "tg-base"; overlay.className = "tg-overlay";
    overlay.tabIndex = 0;
    overlay.setAttribute("role", "img");
    overlay.setAttribute("aria-label", `Transaction graph: ${int(m.n_tx)} transactions linked to ${int(m.n_entities)} accounts, cards, devices, IP addresses and merchants; ${int(m.n_fraud)} ${WD.many} in red. Use the search box, the fraud ring list or the arrow and plus/minus keys to explore.`);
    wrap.append(base, overlay);
    bctx = base.getContext("2d"); octx = overlay.getContext("2d");
    resize();
    fit();
    new ResizeObserver(() => { if (Math.abs(wrap.clientWidth - W) > 2) { resize(); render(); } }).observe(wrap);
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { readColors(); render(); });
    bindInput();
    render();
  }

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = wrap.clientWidth;
    H = Math.round(clamp(W * 0.72, 420, 720));
    for (const c of [base, overlay]) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
      c.style.width = `${W}px`; c.style.height = `${H}px`;
    }
    wrap.style.height = `${H}px`;
    snapshot = null;
  }

  const sx = (x) => (x - view.cx) * view.k + W / 2;
  const sy = (y) => (y - view.cy) * view.k + H / 2;
  const wx = (px) => (px - W / 2) / view.k + view.cx;
  const wy = (py) => (py - H / 2) / view.k + view.cy;

  function fit(bounds) {
    const [x0, y0, x1, y1] = bounds || G.meta.bounds;
    view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
    view.k = Math.min(W / Math.max(x1 - x0, 1e-6), H / Math.max(y1 - y0, 1e-6)) * 0.94;
  }
  const focusOn = (x, y, r) => fit([x - r, y - r, x + r, y + r]);

  // ---------------------------------------------------------------- drawing
  function nodeShape(ctx, shape, x, y, r) {
    switch (shape) {
      case "square": case "hollow-square": ctx.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7); break;
      // no closePath(): with tens of thousands of subpaths it makes path building crawl; an explicit lineTo closes the shape
      case "diamond": ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.15, y); ctx.lineTo(x, y + r * 1.15); ctx.lineTo(x - r * 1.15, y); ctx.lineTo(x, y - r * 1.15); break;
      case "triangle": ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r, y + r * 0.75); ctx.lineTo(x - r, y + r * 0.75); ctx.lineTo(x, y - r * 1.15); break;
      case "plus": ctx.rect(x - r, y - r * 0.3, 2 * r, r * 0.6); ctx.rect(x - r * 0.3, y - r, r * 0.6, 2 * r); break;
      default: ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 2 * Math.PI);
    }
  }
  const entRadius = (e) => clamp(1.6 + Math.sqrt(G.A.ent_deg[e]) * 0.55, 2.4, 9) * clamp(Math.sqrt(view.k / 6), 0.55, 1.7);

  function render() {
    const t0 = performance.now(), mark = {};
    const ctx = bctx, { A, N, T, E } = G, m = G.meta;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.panel;
    ctx.fillRect(0, 0, W, H);
    const sel = selection;
    const inView = (x, y, pad) => x > -pad && x < W + pad && y > -pad && y < H + pad;
    const big = N > 60000;

    // edges: grey for legitimate, red for fraud; hub links fainter
    const edgePass = (wantFraud, wantHub, alpha, color, only) => {
      ctx.beginPath();
      let drawn = 0;
      for (let i = 0; i < N; i++) {
        const fraud = A.tx_flags[i] & 1;
        if ((fraud === 1) !== wantFraud) continue;
        if (!fraud && !showLegit) continue;
        if (only && !only.tx.has(i)) continue;
        const tx = sx(A.tx_xy[2 * i]), ty = sy(A.tx_xy[2 * i + 1]);
        for (let t = 0; t < T; t++) {
          if (G.hub[t] !== wantHub || !visibleType[t]) continue;
          const e = A.tx_ent[i * T + t];
          if (e < 0 || (only && !only.ent.has(e))) continue;
          const ex = sx(A.ent_xy[2 * e]), ey = sy(A.ent_xy[2 * e + 1]);
          if (!inView(tx, ty, 0) && !inView(ex, ey, 0) &&
              ((tx < 0 && ex < 0) || (tx > W && ex > W) || (ty < 0 && ey < 0) || (ty > H && ey > H))) continue;
          ctx.moveTo(tx, ty); ctx.lineTo(ex, ey); drawn++;
        }
      }
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 0.6; ctx.stroke(); ctx.globalAlpha = 1;
      return drawn;
    };
    const dim = sel ? 0.22 : 1;
    edgePass(false, true, (big ? 0.02 : 0.035) * dim, colors.muted);
    edgePass(false, false, (big ? 0.18 : 0.32) * dim, colors.muted);
    edgePass(true, true, 0.25 * dim, colors.fraud);
    edgePass(true, false, 0.6 * dim, colors.fraud);
    mark.edges = performance.now() - t0;
    drawNodes(ctx, null, dim);
    mark.nodes = performance.now() - t0 - mark.edges;
    if (sel) {   // the traced neighbourhood on top, at full strength
      edgePass(false, true, 0.35, colors.muted, sel);
      edgePass(false, false, 0.75, colors.ink, sel);
      edgePass(true, false, 0.9, colors.fraud, sel);
      edgePass(true, true, 0.6, colors.fraud, sel);
      drawNodes(ctx, sel, 1);
    }
    drawLabels(ctx, sel);
    root.dataset.renderMs = JSON.stringify({ total: Math.round(performance.now() - t0), edges: Math.round(mark.edges), nodes: Math.round(mark.nodes) });
    snapshot = null;
    snapView = { ...view };
    drawOverlay();
  }

  function drawNodes(ctx, only, alpha) {
    const { A, N, E, T } = G;
    const pad = 12;
    // legitimate transactions: tiny squares (fast); fraud: red dots on top
    const s = clamp(view.k * 0.11, 1.1, 3.2);
    if (showLegit) {   // one path, one fill: hundreds of thousands of separate fillRect calls are far slower
      ctx.globalAlpha = 0.75 * alpha; ctx.fillStyle = colors.muted;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        if (A.tx_flags[i] & 1 || (only && !only.tx.has(i))) continue;
        const x = sx(A.tx_xy[2 * i]), y = sy(A.tx_xy[2 * i + 1]);
        if (x < -pad || x > W + pad || y < -pad || y > H + pad) continue;
        ctx.rect(x - s / 2, y - s / 2, s, s);
      }
      ctx.fill();
    }
    // entities, one path per type
    ctx.globalAlpha = alpha;
    for (let t = 0; t < T; t++) {
      if (!visibleType[t]) continue;
      const shape = shapeOf(G.meta.types[t]);
      ctx.beginPath();
      const fraudRing = new Path2D();
      for (let e = 0; e < E; e++) {
        if (A.ent_type[e] !== t || (only && !only.ent.has(e))) continue;
        const x = sx(A.ent_xy[2 * e]), y = sy(A.ent_xy[2 * e + 1]);
        if (x < -pad || x > W + pad || y < -pad || y > H + pad) continue;
        const r = entRadius(e);
        nodeShape(ctx, shape, x, y, r);
        if (A.ent_fraud[e] > 0) { fraudRing.moveTo(x + r + 2.2, y); fraudRing.arc(x, y, r + 2.2, 0, 2 * Math.PI); }
      }
      if (shape.startsWith("hollow")) { ctx.lineWidth = 1.4; ctx.strokeStyle = colors.types[t]; ctx.stroke(); }
      else { ctx.fillStyle = colors.types[t]; ctx.fill(); ctx.lineWidth = 0.8; ctx.strokeStyle = colors.panel; ctx.stroke(); }
      ctx.lineWidth = 1.3; ctx.strokeStyle = colors.fraud; ctx.stroke(fraudRing);
    }
    // fraud transactions
    const r = clamp(view.k * 0.16, 1.8, 4.2);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      if (!(A.tx_flags[i] & 1) || (only && !only.tx.has(i))) continue;
      const x = sx(A.tx_xy[2 * i]), y = sy(A.tx_xy[2 * i + 1]);
      if (x < -pad || x > W + pad || y < -pad || y > H + pad) continue;
      ctx.moveTo(x + r, y); ctx.arc(x, y, r, 0, 2 * Math.PI);
    }
    ctx.fillStyle = colors.fraud; ctx.fill();
    ctx.lineWidth = 0.8; ctx.strokeStyle = colors.panel; ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawLabels(ctx, only) {
    // names appear once there is room; most important first (used in fraud, then busiest), never overlapping
    if (view.k < 22 && !only) return;
    const { A, E } = G;
    const cand = [];
    const consider = (e) => {
      if (!visibleType[A.ent_type[e]] || (only && !only.ent.has(e))) return;
      if (G.hub[A.ent_type[e]] && view.k < 40 && !only) return;
      const x = sx(A.ent_xy[2 * e]), y = sy(A.ent_xy[2 * e + 1]);
      if (x < 0 || x > W || y < 0 || y > H) return;
      cand.push(e);
    };
    if (only) only.ent.forEach(consider); else for (let e = 0; e < E; e++) consider(e);
    cand.sort((a, b) => A.ent_fraud[b] - A.ent_fraud[a] || A.ent_deg[b] - A.ent_deg[a]);
    ctx.font = "11px Figtree, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const cell = 10, cols = Math.ceil(W / cell) + 1, taken = new Uint8Array(cols * (Math.ceil(H / cell) + 1));
    let placed = 0;
    for (const e of cand) {
      if (placed >= 250) break;
      const name = G.meta.names[e];
      const x = sx(A.ent_xy[2 * e]) + entRadius(e) + 4, y = sy(A.ent_xy[2 * e + 1]);
      const w = ctx.measureText(name).width;
      const c0 = Math.floor(x / cell), c1 = Math.floor((x + w) / cell), r0 = Math.floor((y - 6) / cell), r1 = Math.floor((y + 6) / cell);
      if (c0 < 0 || r0 < 0 || c1 >= cols || (r1 + 1) * cols > taken.length) continue;
      let free = true;
      for (let r = r0; r <= r1 && free; r++) for (let c = c0; c <= c1; c++) if (taken[r * cols + c]) { free = false; break; }
      if (!free) continue;
      for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) taken[r * cols + c] = 1;
      ctx.lineWidth = 3; ctx.strokeStyle = colors.panel; ctx.strokeText(name, x, y);   // halo keeps text legible over edges
      ctx.fillStyle = A.ent_fraud[e] ? colors.fraud : colors.ink; ctx.fillText(name, x, y);
      placed++;
    }
  }

  function drawOverlay() {
    const ctx = octx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const ring = (id, color, width) => {
      if (id < 0) return;
      const { x, y, r } = nodeGeom(id);
      ctx.beginPath(); ctx.arc(x, y, r + 4, 0, 2 * Math.PI);
      ctx.lineWidth = width; ctx.strokeStyle = color; ctx.stroke();
    };
    if (selection && selection.id >= 0) ring(selection.id, colors.ink, 2.5);
    ring(hoverId, colors.ink, 1.5);
  }

  function nodeGeom(id) {
    const { A, N } = G;
    if (id < N) return { x: sx(A.tx_xy[2 * id]), y: sy(A.tx_xy[2 * id + 1]), r: A.tx_flags[id] & 1 ? clamp(view.k * 0.16, 1.8, 4.2) : 2 };
    const e = id - N;
    return { x: sx(A.ent_xy[2 * e]), y: sy(A.ent_xy[2 * e + 1]), r: entRadius(e) };
  }

  // fast redraw while the view moves: scale/translate the last full render, full detail once idle
  function quickDraw() {
    if (!snapView) return render();
    if (!snapshot) { snapshot = document.createElement("canvas"); snapshot.width = base.width; snapshot.height = base.height; snapshot.getContext("2d").drawImage(base, 0, 0); }
    const s = view.k / snapView.k;
    const ox = (snapView.cx - view.cx) * view.k + W / 2 - (W / 2) * s;
    const oy = (snapView.cy - view.cy) * view.k + H / 2 - (H / 2) * s;
    bctx.setTransform(1, 0, 0, 1, 0, 0);
    bctx.fillStyle = colors.panel; bctx.fillRect(0, 0, base.width, base.height);
    bctx.setTransform(dpr * s, 0, 0, dpr * s, ox * dpr, oy * dpr);
    bctx.drawImage(snapshot, 0, 0, base.width / dpr, base.height / dpr);
    hoverId = -1; drawOverlay();
    clearTimeout(idleTimer);
    idleTimer = setTimeout(render, 140);
  }

  // ---------------------------------------------------------------- picking
  function pick(px, py) {
    const { A, N, grid } = G;
    const x = wx(px), y = wy(py), tol = 9 / view.k;
    const c0 = Math.floor((x - tol - grid.x0) / grid.cell), c1 = Math.floor((x + tol - grid.x0) / grid.cell);
    const r0 = Math.floor((y - tol - grid.y0) / grid.cell), r1 = Math.floor((y + tol - grid.y0) / grid.cell);
    let best = -1, bestD = Infinity;
    for (let r = Math.max(0, r0); r <= Math.min(grid.gh - 1, r1); r++) {
      for (let c = Math.max(0, c0); c <= Math.min(grid.gw - 1, c1); c++) {
        const cell = r * grid.gw + c;
        for (let k = grid.start[cell]; k < grid.start[cell + 1]; k++) {
          const id = grid.ids[k];
          let nx, ny, rad;
          if (id < N) {
            if (!showLegit && !(A.tx_flags[id] & 1)) continue;
            nx = A.tx_xy[2 * id]; ny = A.tx_xy[2 * id + 1]; rad = 3;
          } else {
            const e = id - N;
            if (!visibleType[A.ent_type[e]]) continue;
            nx = A.ent_xy[2 * e]; ny = A.ent_xy[2 * e + 1]; rad = entRadius(e) + 2;
          }
          const d = Math.hypot((nx - x) * view.k, (ny - y) * view.k);
          if (d <= Math.max(rad, 6) && d - rad < bestD) { bestD = d - rad; best = id; }
        }
      }
    }
    return best;
  }

  const tip = document.createElement("div");
  tip.className = "chart-tip"; tip.hidden = true;
  document.body.appendChild(tip);
  function showTip(html, evt) {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - 14;
    tip.style.left = `${Math.max(8, x)}px`; tip.style.top = `${Math.max(8, y)}px`;
  }

  function describe(id) {
    const { A, N, meta } = G;
    if (id < N) {
      const f = A.tx_flags[id], split = ["training", "validation", "test"][(f >> 1) & 3];
      const score = meta.has_score ? ` · GNN score ${(A.tx_score[id] / 255).toFixed(2)}` : "";
      return `<b>Transaction ${int(id + 1)}</b><div>${f & 1 ? `<span class="tg-fraud">${WD.yes}</span>` : WD.no}${FLAGS ? "" : ` · ${split} period`}${score}</div><div class="muted">Click to trace its links</div>`;
    }
    const e = id - N, t = A.ent_type[e];
    const accounts = anchorsOf(e);
    return `<b>${esc(meta.type_labels[t])} ${esc(meta.names[e])}</b><div>${int(A.ent_deg[e])} transaction${A.ent_deg[e] === 1 ? "" : "s"}${accounts > 1 ? ` on ${int(accounts)} ${esc(meta.type_labels[meta.types.indexOf(meta.anchor_type)] || "account").toLowerCase()}s` : ""}</div>` +
      (A.ent_fraud[e] ? `<div class="tg-fraud">${int(A.ent_fraud[e])} ${WD.adj}</div>` : "") + `<div class="muted">Click to trace its links</div>`;
  }

  function anchorsOf(e) {
    const { A, T, meta } = G, at = meta.types.indexOf(meta.anchor_type);
    if (at < 0 || G.meta.types[A.ent_type[e]] === meta.anchor_type) return 1;
    const seen = new Set();
    for (let k = G.entOff[e]; k < G.entOff[e + 1] && seen.size < 10000; k++) {
      const a = A.tx_ent[G.entTx[k] * T + at];
      if (a >= 0) seen.add(a);
    }
    return seen.size;
  }

  // ---------------------------------------------------------------- selection (trace links)
  const CAP = 6000;
  function neighbourhood(id) {
    const { A, N, T } = G;
    const tx = new Set(), ent = new Set();
    const addTxOf = (e) => { for (let k = G.entOff[e]; k < G.entOff[e + 1] && tx.size < CAP; k++) tx.add(G.entTx[k]); };
    const addEntsOf = (i) => { for (let t = 0; t < T; t++) { const e = A.tx_ent[i * T + t]; if (e >= 0 && visibleType[t]) ent.add(e); } };
    if (id < N) {
      tx.add(id); addEntsOf(id);
      for (const e of [...ent]) if (!G.hub[A.ent_type[e]]) addTxOf(e);   // 2 hops through identity entities only
    } else {
      const e0 = id - N;
      ent.add(e0); addTxOf(e0);
      if (!G.hub[A.ent_type[e0]]) for (const i of [...tx]) addEntsOf(i);
    }
    for (const i of tx) if (ent.size < CAP) addEntsOf(i);
    return { id, tx, ent };
  }

  function select(id, { zoom = false } = {}) {
    if (id < 0) { selection = null; panel.innerHTML = panelHint(); render(); return; }
    selection = neighbourhood(id);
    if (zoom) {
      const { x, y } = worldOf(id);
      focusOn(x, y, 6);
    }
    render();
    showDetails(id);
  }

  const worldOf = (id) => id < G.N ? { x: G.A.tx_xy[2 * id], y: G.A.tx_xy[2 * id + 1] } : { x: G.A.ent_xy[2 * (id - G.N)], y: G.A.ent_xy[2 * (id - G.N) + 1] };
  const panelHint = () => `<p class="small muted">Hover a node for a summary. Click one to trace everything linked to it: its transactions and the accounts, cards, devices and IPs they share.</p>`;

  async function showDetails(id) {
    const { A, N, T, meta } = G;
    const s = selection;
    const nFraud = [...s.tx].filter((i) => A.tx_flags[i] & 1).length;
    const head = `<p class="small muted tg-trace">Traced ${int(s.tx.size)} transaction${s.tx.size === 1 ? "" : "s"} (${int(nFraud)} ${WD.label}) and ${int(s.ent.size)} entities${s.tx.size >= CAP ? ", capped" : ""}.</p>`;
    const entButton = (e) => `<button type="button" class="linklike" data-ent="${e}">${esc(meta.names[e])}</button>${A.ent_fraud[e] ? ' <span class="tg-fraud">●</span>' : ""}`;
    if (id >= N) {
      const e = id - N, t = A.ent_type[e];
      const linked = {};
      for (const x of s.ent) if (x !== e) (linked[A.ent_type[x]] ||= []).push(x);
      const groups = Object.entries(linked).map(([tt, list]) => {
        list.sort((a, b) => A.ent_fraud[b] - A.ent_fraud[a] || A.ent_deg[b] - A.ent_deg[a]);
        return `<dt>${esc(plural(meta.type_labels[tt]))} (${int(list.length)})</dt><dd>${list.slice(0, 8).map(entButton).join(", ")}${list.length > 8 ? ` and ${int(list.length - 8)} more` : ""}</dd>`;
      }).join("");
      panel.innerHTML = `<h4>${esc(meta.type_labels[t])} <span class="tg-name">${esc(meta.names[e])}</span></h4>
        <p>${int(A.ent_deg[e])} transactions, <b class="${A.ent_fraud[e] ? "tg-fraud" : ""}">${int(A.ent_fraud[e])} ${WD.adj}</b>.</p>${head}<dl class="tg-linked">${groups}</dl>`;
    } else {
      panel.innerHTML = `<h4>Transaction ${int(id + 1)}</h4><p class="small muted">Loading…</p>`;
      try {
        const d = await (await fetch(root.dataset.tx.replace(/0\/$/, `${id}/`))).json();
        if (selection !== s) return;
        panel.innerHTML = `<h4>Transaction <span class="tg-name">${esc(d.id)}</span></h4>
          <dl class="tg-kv">
            <dt>${WD.verdict}</dt><dd>${d.fraud ? `<b class="tg-fraud">${WD.yes}</b>` : WD.no}</dd>
            ${d.amount != null ? `<dt>Amount</dt><dd>${d.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</dd>` : ""}
            ${d.time ? `<dt>Time</dt><dd>${esc(d.time)}</dd>` : ""}
            ${FLAGS ? "" : `<dt>Period</dt><dd>${esc(d.split)}</dd>`}
            ${d.gnn_score != null ? `<dt>GNN score</dt><dd>${d.gnn_score.toFixed(2)}${d.gnn_flagged ? " · above its alert threshold" : ""}</dd>` : ""}
          </dl>${head}
          <dl class="tg-linked">${d.entities.map((x) => `<dt>${esc(x.label)}</dt><dd>${entButton(x.index)}</dd>`).join("")}</dl>`;
      } catch { panel.innerHTML = `<h4>Transaction ${int(id + 1)}</h4>${head}`; }
    }
  }
  panel.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-ent]");
    if (b) select(G.N + Number(b.dataset.ent), { zoom: true });
  });

  // ---------------------------------------------------------------- controls
  function buildControls() {
    const m = G.meta, box = $(".tg-filters");
    const icon = (t, i) => {
      const shape = shapeOf(t), c = `var(${typeColorVar(t, i)})`;
      const hollow = shape.startsWith("hollow") || shape === "plus";
      const d = { circle: '<circle cx="7" cy="7" r="5"/>', "hollow-circle": '<circle cx="7" cy="7" r="4.5"/>', square: '<rect x="2.5" y="2.5" width="9" height="9"/>',
        "hollow-square": '<rect x="2.5" y="2.5" width="9" height="9"/>', diamond: '<path d="M7 1.5 12.5 7 7 12.5 1.5 7Z"/>',
        triangle: '<path d="M7 1.5 12.5 11.5H1.5Z"/>', plus: '<path d="M2 7H12M7 2V12"/>' }[shape];
      return `<svg width="14" height="14" aria-hidden="true" style="${hollow ? `fill:none;stroke:${c};stroke-width:1.6` : `fill:${c}`}">${d}</svg>`;
    };
    box.innerHTML = m.types.map((t, i) => `<label><input type="checkbox" data-type="${i}" ${visibleType[i] ? "checked" : ""}> ${icon(t, i)} ${esc(plural(m.type_labels[i]))} <span class="muted">${int(m.entities_by_type[t])}</span></label>`).join("") +
      `<label><input type="checkbox" data-legit checked> <svg width="14" height="14" aria-hidden="true"><rect x="5" y="5" width="4" height="4" style="fill:var(--muted)"/></svg> ${WD.others}</label>` +
      `<span class="tg-key"><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4" style="fill:var(--fraud)"/></svg> ${WD.key}</span>` +
      `<span class="tg-key"><svg width="14" height="14" aria-hidden="true"><circle cx="7" cy="7" r="4.5" style="fill:none;stroke:var(--fraud);stroke-width:1.5"/></svg> ${WD.entKey}</span>`;
    box.addEventListener("change", (ev) => {
      const el = ev.target;
      if (el.dataset.type != null) visibleType[Number(el.dataset.type)] = el.checked;
      if (el.hasAttribute("data-legit")) showLegit = el.checked;
      if (selection && selection.id >= 0) selection = neighbourhood(selection.id);
      render();
    });
    if (m.hub_types.length && G.N > 60000) {
      $(".tg-note").textContent = `${m.hub_types.map((t) => m.type_labels[m.types.indexOf(t)]).join(" and ")} links are hidden at first on a file this size, because nearly every account shares them. Tick the box to draw them.`;
    }
    $(".tg-fit").addEventListener("click", () => { fit(); render(); });
    $(".tg-in").addEventListener("click", () => zoomAt(W / 2, H / 2, 1.6));
    $(".tg-out").addEventListener("click", () => zoomAt(W / 2, H / 2, 1 / 1.6));
    $(".tg-search").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const q = $(".tg-search input").value.trim().toLowerCase();
      const out = $(".tg-search-msg");
      if (!q) return;
      let hit = G.lower.indexOf(q);
      if (hit < 0) hit = G.lower.findIndex((s) => s.startsWith(q));
      if (hit < 0) hit = G.lower.findIndex((s) => s.includes(q));
      if (hit < 0) { out.textContent = "No account, card, device, IP or merchant matches that."; return; }
      out.textContent = "";
      const t = G.A.ent_type[hit];
      if (!visibleType[t]) { visibleType[t] = true; $(`[data-type="${t}"]`).checked = true; }
      select(G.N + hit, { zoom: true });
    });
    panel.innerHTML = panelHint();
  }

  function buildRings() {
    const m = G.meta, list = $(".tg-rings");
    const anchor = m.anchor_type ? m.type_labels[m.types.indexOf(m.anchor_type)].toLowerCase() : "account";
    if (!m.rings.length) {
      list.innerHTML = `<p class="small muted">No ${WD.many} transactions on different ${esc(anchor)}s share a device, IP address or card.</p>`;
      return;
    }
    const rings = m.rings.filter((r) => Array.isArray(r.tx) && Array.isArray(r.ents));
    if (!rings.length) { list.innerHTML = `<p class="small muted">Retrain this dataset to list its fraud rings.</p>`; return; }
    m.rings = rings;
    const evidence = (r) => (r.shared || []).length
      ? r.shared.slice(0, 2).map((s) => `${esc(m.type_labels[m.types.indexOf(s.type)])} <span class="tg-name">${esc(s.name)}</span>`).join(", ")
      : "linked through their cards";
    list.innerHTML = `<ol>${m.rings.slice(0, 12).map((r, i) => `<li><button type="button" class="linklike" data-ring="${i}">${int(r.anchors)} ${esc(anchor)}s, ${int(r.frauds)} ${WD.many}</button><div class="small muted">sharing ${evidence(r)}</div></li>`).join("")}</ol>` +
      (m.n_rings > 12 ? `<p class="small muted">and ${int(m.n_rings - 12)} more.</p>` : "");
    list.addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-ring]");
      if (!b) return;
      const r = m.rings[Number(b.dataset.ring)];
      for (const e of r.ents) { const t = G.A.ent_type[e]; if (!visibleType[t]) { visibleType[t] = true; const cb = $(`[data-type="${t}"]`); if (cb) cb.checked = true; } }
      selection = { id: -1, tx: new Set(r.tx), ent: new Set(r.ents) };
      focusOn(r.x, r.y, Math.max(r.r * 1.4, 5));
      render();
      const shared = (r.shared || []).map((s) => `<li>${esc(m.type_labels[m.types.indexOf(s.type)])} <span class="tg-name">${esc(s.name)}</span>, used ${WD.usedIn} on ${int(s.accounts)} ${esc(anchor)}s</li>`).join("");
      panel.innerHTML = `<h4>${WD.ring}</h4><p>${int(r.frauds)} ${WD.adj} transactions on ${int(r.anchors)} ${esc(anchor)}s, joined through the entities they share.</p>` +
        (shared ? `<ul class="tg-shared">${shared}</ul>` : "") + `<p class="small muted">Only the ${WD.adj} transactions are highlighted. Click any node for its full history.</p>`;
    });
  }

  // ---------------------------------------------------------------- input
  function zoomAt(px, py, f) {
    const x = wx(px), y = wy(py);
    view.k = clamp(view.k * f, 0.02, 4000);
    view.cx = x - (px - W / 2) / view.k; view.cy = y - (py - H / 2) / view.k;
    quickDraw();
  }

  function bindInput() {
    const el = overlay, pts = new Map();
    let drag = null, moved = false, raf = 0, lastEvt = null;
    const local = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    el.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = local(e); zoomAt(x, y, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId); pts.set(e.pointerId, local(e)); moved = false;
      if (pts.size === 1) drag = { p: local(e), cx: view.cx, cy: view.cy };
      else drag = null;
    });
    el.addEventListener("pointermove", (e) => {
      const p = local(e);
      if (pts.has(e.pointerId)) {
        const prev = pts.get(e.pointerId); pts.set(e.pointerId, p);
        if (pts.size === 2) {   // pinch
          const [a, b] = [...pts.values()], other = [...pts.entries()].find(([k]) => k !== e.pointerId)[1];
          const d0 = Math.hypot(prev[0] - other[0], prev[1] - other[1]), d1 = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (d0 > 0) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d1 / d0);
          moved = true; return;
        }
        if (drag && Math.hypot(p[0] - drag.p[0], p[1] - drag.p[1]) > 3) {
          moved = true; tip.hidden = true;
          view.cx = drag.cx - (p[0] - drag.p[0]) / view.k; view.cy = drag.cy - (p[1] - drag.p[1]) / view.k;
          quickDraw();
        }
        return;
      }
      lastEvt = e;
      if (!raf) raf = requestAnimationFrame(() => {
        raf = 0;
        const [x, y] = local(lastEvt), id = pick(x, y);
        if (id !== hoverId) { hoverId = id; drawOverlay(); }
        if (id >= 0) { showTip(describe(id), lastEvt); el.style.cursor = "pointer"; } else { tip.hidden = true; el.style.cursor = "grab"; }
      });
    });
    const up = (e) => {
      const was = pts.has(e.pointerId);
      pts.delete(e.pointerId);
      if (!was) return;
      if (!moved && pts.size === 0) { const [x, y] = local(e); select(pick(x, y)); }
      drag = null;
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("pointerleave", () => { tip.hidden = true; if (hoverId >= 0) { hoverId = -1; drawOverlay(); } });
    el.addEventListener("keydown", (e) => {
      const step = 60 / view.k;
      const act = { "+": () => zoomAt(W / 2, H / 2, 1.4), "=": () => zoomAt(W / 2, H / 2, 1.4), "-": () => zoomAt(W / 2, H / 2, 1 / 1.4),
        ArrowLeft: () => { view.cx -= step; quickDraw(); }, ArrowRight: () => { view.cx += step; quickDraw(); },
        ArrowUp: () => { view.cy -= step; quickDraw(); }, ArrowDown: () => { view.cy += step; quickDraw(); },
        "0": () => { fit(); render(); }, Escape: () => select(-1) }[e.key];
      if (act) { e.preventDefault(); act(); }
    });
  }
})();
