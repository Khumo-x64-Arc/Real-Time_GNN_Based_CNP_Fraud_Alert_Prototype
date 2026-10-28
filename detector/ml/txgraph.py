"""Lay out the whole transaction graph (the one the GNN learns on) for drawing.

The graph is bipartite: every transaction is a node linked to each entity it
touches (account, card, device, IP, merchant, terminal). That is exactly the
GNN's message-passing structure (transaction -> entity -> transaction), and it
stays readable where a transaction-to-transaction graph would not: a device
used 50 times is 50 edges here, not 1,225.

Layout (computed once at training time, so the page never waits on physics):

* Identity entities (account, card, device, IP, or whatever non-hub types the
  file has) form connected clusters. Usually that is one account with its
  card, devices and IPs; a cluster spanning several accounts means they share
  a device or IP, which is what fraud rings look like.
* Small single-account clusters use a fixed star (account in the middle).
  Larger ones get a force-directed (Fruchterman-Reingold) layout. Very large
  ones (shared NAT IPs can chain thousands of accounts together) are
  coarsened first: entities only one account uses orbit that account, and
  the shared skeleton is laid out with a particle-mesh force layout
  (repulsion from grid-cell centroids instead of every pair), which scales
  to tens of thousands of nodes.
* Clusters are packed largest-first on a sunflower spiral.
* Hub entities (merchants, terminals) touch huge numbers of accounts, so they
  sit on an outer ring, ordered by where their customers are.
* Each transaction is drawn at the centre of its identity entities with a
  small deterministic jitter, so an account's transactions form a cloud
  around it.

Fraud rings are found on the fraud-only subgraph: fraudulent transactions and
the identity entities they used. A connected piece of it that spans two or
more accounts means those accounts' frauds share a device, IP or card.

Output: a JSON header (names, counts, stats, fraud rings) and one binary blob
of typed arrays that graph.js reads directly.
"""
from __future__ import annotations

import gzip
import json
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HUB_TYPES = {"merchant", "terminal"}
TYPE_LABELS = {"account": "Account", "card": "Card", "device": "Device", "ip": "IP address",
               "merchant": "Merchant", "terminal": "Terminal", "amount_bucket": "Amount bucket", "none": "None"}
STAR_MAX = 12          # clusters this small with a single anchor use the fixed star layout
FR_MAX = 1500          # largest cluster laid out by force; bigger ones use BFS rings
TX_JITTER = 0.42       # transaction cloud radius around its entities (entity spacing = 1)
GOLDEN = math.pi * (3 - math.sqrt(5))


def _fr_layout(n, edges, w, rng, iters=120):
    """Fruchterman-Reingold for one cluster (n <= FR_MAX), vectorised over all pairs."""
    pos = rng.normal(0, 1, (n, 2)) * math.sqrt(n) * 0.5
    k = 1.0
    t = math.sqrt(n) * 0.5
    src, dst = edges[:, 0], edges[:, 1]
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist2 = (d ** 2).sum(-1) + 1e-6
        disp = (d * (k * k / dist2)[:, :, None]).sum(1)                     # repulsion
        e = pos[src] - pos[dst]
        el = np.sqrt((e ** 2).sum(1)) + 1e-9
        f = (e / el[:, None]) * (el * el / k * w)[:, None]                 # attraction
        np.add.at(disp, src, -f)
        np.add.at(disp, dst, f)
        dl = np.sqrt((disp ** 2).sum(1)) + 1e-9
        pos += disp / dl[:, None] * np.minimum(dl, t)[:, None]
        t *= 0.96
    pos -= pos.mean(0)
    # rescale so the median edge is about 1 (same spacing as the star layout)
    el = np.sqrt(((pos[src] - pos[dst]) ** 2).sum(1)) if len(src) else np.ones(1)
    return pos / max(np.median(el), 1e-6)


def _pm_layout(n, edges, w, rng, iters=90, grid=40, chunk=4096):
    """Force layout for big graphs: exact spring attraction, particle-mesh repulsion."""
    pos = rng.normal(0, 1, (n, 2)) * math.sqrt(n) * 0.6
    k, t = 1.0, math.sqrt(n) * 0.5
    src, dst = edges[:, 0], edges[:, 1]
    for _ in range(iters):
        lo = pos.min(0)
        cell = max((pos.max(0) - lo).max() / grid, 1e-6)
        g = np.clip(((pos - lo) / cell).astype(int), 0, grid - 1)
        cid = g[:, 1] * grid + g[:, 0]
        mass = np.bincount(cid, minlength=grid * grid).astype(float)
        nz = mass > 0
        C = np.c_[np.bincount(cid, pos[:, 0], grid * grid)[nz], np.bincount(cid, pos[:, 1], grid * grid)[nz]] / mass[nz, None]
        m = mass[nz]
        soft = (0.5 * cell) ** 2 + k * k
        disp = np.empty_like(pos)
        for s in range(0, n, chunk):
            d = pos[s:s + chunk, None, :] - C[None, :, :]
            disp[s:s + chunk] = (d * (k * k * m / ((d ** 2).sum(-1) + soft))[:, :, None]).sum(1)
        e = pos[src] - pos[dst]
        el = np.sqrt((e ** 2).sum(1)) + 1e-9
        f = (e / el[:, None]) * (el * el / k * w)[:, None]
        np.add.at(disp, src, -f)
        np.add.at(disp, dst, f)
        dl = np.sqrt((disp ** 2).sum(1)) + 1e-9
        pos += disp / dl[:, None] * np.minimum(dl, t)[:, None]
        t = max(t * 0.95, 0.05)
    pos -= pos.mean(0)
    el = np.sqrt(((pos[src] - pos[dst]) ** 2).sum(1)) if len(src) else np.ones(1)
    return pos / max(np.median(el), 1e-6)


def _large_layout(members, ent_type, anchor_ti, ed, w, rng):
    """Coarsen (private entities orbit their only account), lay out the skeleton, re-expand."""
    k = len(members)
    is_anchor = ent_type[members] == anchor_ti
    # which anchors each entity is linked to (edges are anchor-other when an anchor exists)
    a_side = np.where(is_anchor[ed[:, 0]], ed[:, 0], np.where(is_anchor[ed[:, 1]], ed[:, 1], -1))
    o_side = np.where(a_side == ed[:, 0], ed[:, 1], ed[:, 0])
    has_a = a_side >= 0
    n_anchor = np.zeros(k, dtype=int)
    owner = np.full(k, -1)
    pairs = np.unique(np.c_[o_side[has_a], a_side[has_a]], axis=0)
    np.add.at(n_anchor, pairs[:, 0], 1)
    owner[pairs[:, 0]] = pairs[:, 1]
    private = (~is_anchor) & (n_anchor == 1)
    core = np.where(~private)[0]
    remap = np.full(k, -1)
    remap[core] = np.arange(len(core))
    # skeleton edges: private entities are folded into their account
    end = np.where(private, owner, np.arange(k))
    ce = remap[end[ed]]
    keep = ce[:, 0] != ce[:, 1]
    ce, cw = ce[keep], w[keep]
    if len(ce):
        ce, inv = np.unique(np.sort(ce, 1), axis=0, return_inverse=True)
        cw = np.bincount(inv.ravel(), weights=cw)
    local = np.zeros((k, 2))
    core_pos = (_fr_layout(len(core), ce, cw / max(cw.mean(), 1e-9), rng, iters=60) if len(core) <= FR_MAX
                else _pm_layout(len(core), ce, cw / max(cw.mean(), 1e-9), rng))
    local[core] = core_pos * 3.0   # room for each account's own devices and IPs
    for a in np.unique(owner[private]):
        sats = np.where(private & (owner == a))[0]
        ang = np.arange(len(sats)) * 2 * math.pi / len(sats) + rng.random()
        local[sats] = local[a] + np.c_[np.cos(ang), np.sin(ang)]
    return local


def _bfs_layout(n, edges):
    """Concentric rings by hop distance from the best-connected node (for huge clusters)."""
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    root = int(np.asarray(A.sum(1)).ravel().argmax())
    from scipy.sparse.csgraph import breadth_first_order
    order, pred = breadth_first_order(A, root, directed=False)
    depth = np.zeros(n, dtype=int)
    for v in order[1:]:
        depth[v] = depth[pred[v]] + 1
    pos = np.zeros((n, 2))
    for dd in range(1, depth.max() + 1):
        ring = np.where(depth == dd)[0]
        r = max(dd * 1.5, len(ring) / (2 * math.pi) * 0.9)
        ang = np.linspace(0, 2 * math.pi, len(ring), endpoint=False) + dd
        pos[ring] = np.c_[np.cos(ang), np.sin(ang)] * r
    return pos


def build_graph(keys: dict[str, list], y: np.ndarray, split: np.ndarray, score: np.ndarray | None,
                threshold: float | None, seed: int = 7):
    """keys[type][i] = entity key of transaction i (or None). split: 0 train, 1 validation, 2 test.

    Returns (meta: dict, blob: bytes) — blob is gzip-compressed typed arrays described in meta["arrays"].
    """
    rng = np.random.default_rng(seed)
    types = list(keys)
    n = len(y)
    T = len(types)

    # ---- entities
    ent_index: dict = {}
    ent_type, ent_name = [], []
    tx_ent = np.full((n, T), -1, dtype=np.int32)
    for ti, t in enumerate(types):
        col = tx_ent[:, ti]
        for i, k in enumerate(keys[t]):
            if k is None:
                continue
            j = ent_index.get(k)
            if j is None:
                j = ent_index[k] = len(ent_type)
                ent_type.append(ti)
                ent_name.append(str(k).split(":", 1)[-1])
            col[i] = j
    E = len(ent_type)
    ent_type = np.asarray(ent_type, dtype=np.uint8)
    rows, slots = np.nonzero(tx_ent >= 0)
    ents = tx_ent[rows, slots]
    deg = np.bincount(ents, minlength=E).astype(np.uint32)
    fraud_deg = np.bincount(ents, weights=(y[rows] == 1), minlength=E).astype(np.uint32)

    hub_types = [t for t in types if t in HUB_TYPES]
    struct_types = [t for t in types if t not in HUB_TYPES] or types
    hub_types = [t for t in types if t not in struct_types]
    struct_ti = [types.index(t) for t in struct_types]
    anchor_ti = struct_ti[0]   # account if present, else the first identity type
    is_struct = np.isin(ent_type, struct_ti)

    # ---- clusters: connected components of the transaction <-> identity-entity graph
    m = is_struct[ents]
    B = sp.csr_matrix((np.ones(m.sum()), (rows[m], ents[m])), shape=(n, E))
    G = sp.bmat([[None, B], [B.T, None]]).tocsr()
    n_comp, comp = connected_components(G, directed=False)
    ent_comp = comp[n:]
    tx_comp = comp[:n]

    # entity-entity layout edges: a transaction links its anchor (account) to its other identity entities;
    # without an anchor it chains its identity entities in type order
    has_anchor = tx_ent[:, anchor_ti] >= 0
    pairs = []
    others = [ti for ti in struct_ti if ti != anchor_ti]
    for b in others:
        ok = has_anchor & (tx_ent[:, b] >= 0)
        pairs.append(np.c_[tx_ent[ok, anchor_ti], tx_ent[ok, b]])
    for a, b in zip(others, others[1:]):
        ok = ~has_anchor & (tx_ent[:, a] >= 0) & (tx_ent[:, b] >= 0)
        pairs.append(np.c_[tx_ent[ok, a], tx_ent[ok, b]])
    pairs = np.vstack(pairs) if pairs else np.zeros((0, 2), dtype=np.int32)
    if len(pairs):
        uniq, cnt = np.unique(np.sort(pairs, 1), axis=0, return_counts=True)
    else:
        uniq, cnt = pairs, np.zeros(0)

    ent_xy = np.zeros((E, 2))
    struct_ents = np.where(is_struct)[0]
    by_comp = {}
    order = np.argsort(ent_comp[struct_ents], kind="stable")
    se = struct_ents[order]
    splits = np.flatnonzero(np.diff(ent_comp[se])) + 1
    for group in np.split(se, splits):
        if len(group):
            by_comp[int(ent_comp[group[0]])] = group
    edge_comp = ent_comp[uniq[:, 0]] if len(uniq) else np.zeros(0, dtype=int)
    e_order = np.argsort(edge_comp, kind="stable")
    e_sorted, w_sorted, ec_sorted = uniq[e_order], cnt[e_order], edge_comp[e_order]
    e_bounds = {}
    if len(ec_sorted):
        starts = np.r_[0, np.flatnonzero(np.diff(ec_sorted)) + 1]
        ends = np.r_[starts[1:], len(ec_sorted)]
        e_bounds = {int(ec_sorted[s]): (s, e) for s, e in zip(starts, ends)}

    comp_layout = []   # (component id, members, local positions, radius)
    for c, members in by_comp.items():
        k = len(members)
        anchors = members[ent_type[members] == anchor_ti]
        if k == 1:
            local = np.zeros((1, 2))
        elif k <= STAR_MAX and len(anchors) <= 1:
            centre = anchors[0] if len(anchors) else members[np.argmax(deg[members])]
            others = members[members != centre]
            others = others[np.lexsort((-deg[others], ent_type[others]))]
            ang = np.arange(len(others)) * 2 * math.pi / len(others) + 0.3
            local = np.zeros((k, 2))
            pos_of = {centre: (0.0, 0.0)}
            for o, a in zip(others, ang):
                pos_of[o] = (math.cos(a), math.sin(a))
            local = np.array([pos_of[mm] for mm in members])
        else:
            s, e = e_bounds.get(c, (0, 0))
            loc = {g: i for i, g in enumerate(members)}
            ed = np.array([[loc[a], loc[b]] for a, b in e_sorted[s:e]], dtype=int).reshape(-1, 2)
            w = np.log1p(w_sorted[s:e]).astype(float)
            if k <= FR_MAX:
                local = _fr_layout(k, ed, w / max(w.mean(), 1e-9), rng, iters=120 if k < 300 else 60)
            elif len(anchors):
                local = _large_layout(members, ent_type, anchor_ti, ed, w, rng)
            else:
                local = _bfs_layout(k, ed)
        radius = float(np.sqrt((local ** 2).sum(1)).max()) + 1.0
        comp_layout.append((c, members, local, radius))

    # ---- pack clusters on a sunflower spiral, biggest in the middle
    comp_layout.sort(key=lambda r: -r[3])
    cum = 0.0
    for i, (c, members, local, radius) in enumerate(comp_layout):
        cum += (radius + 0.35) ** 2
        dist = 0.0 if i == 0 else math.sqrt(cum) * 1.05
        ang = i * GOLDEN
        ent_xy[members] = local + [dist * math.cos(ang), dist * math.sin(ang)]
    core_r = float(np.sqrt((ent_xy[is_struct] ** 2).sum(1)).max()) + 2 if is_struct.any() else 10.0

    # ---- transactions: centre of their identity entities + jitter
    struct_slots = np.array(struct_ti)
    te = tx_ent[:, struct_slots]
    has = te >= 0
    nh = has.sum(1)
    tx_xy = np.zeros((n, 2))
    for j in range(te.shape[1]):
        sel = has[:, j]
        tx_xy[sel] += ent_xy[te[sel, j]]
    linked = nh > 0
    tx_xy[linked] /= nh[linked, None]
    r = TX_JITTER * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * math.pi
    tx_xy += np.c_[np.cos(a), np.sin(a)] * r[:, None]
    unlinked = np.where(~linked)[0]   # transactions with no identity entity: a grid block under the core
    if len(unlinked):
        side = int(math.ceil(math.sqrt(len(unlinked))))
        gx, gy = np.arange(len(unlinked)) % side, np.arange(len(unlinked)) // side
        step = (2 * core_r) / max(side, 1)
        tx_xy[unlinked] = np.c_[-core_r + gx * step, core_r + 3 + gy * step]

    # ---- hubs on an outer ring, ordered by where their transactions are
    hub_ents = np.where(~is_struct)[0]
    if len(hub_ents):
        hm = ~is_struct[ents]
        hs = np.bincount(ents[hm], weights=tx_xy[rows[hm], 0], minlength=E)[hub_ents]
        hc = np.bincount(ents[hm], weights=tx_xy[rows[hm], 1], minlength=E)[hub_ents]
        ang = np.arctan2(hc, hs)
        order_h = np.argsort(ang, kind="stable")
        ring_r = core_r * 1.12 + 2
        even = np.linspace(-math.pi, math.pi, len(hub_ents), endpoint=False)
        ent_xy[hub_ents[order_h]] = np.c_[np.cos(even), np.sin(even)] * ring_r

    # ---- fraud rings: pieces of the fraud-only subgraph (fraud transactions + their identity
    # entities) that span two or more accounts. Only meaningful when the anchor is a real identity.
    rings = []
    fraud_tx = y == 1
    if fraud_tx.any() and types[anchor_ti] in {"account", "card"}:
        fr = np.where(fraud_tx)[0]
        fm = m & fraud_tx[rows]
        Bf = sp.csr_matrix((np.ones(fm.sum()), (np.searchsorted(fr, rows[fm]), ents[fm])), shape=(len(fr), E))
        nf, fcomp = connected_components(sp.bmat([[None, Bf], [Bf.T, None]]).tocsr(), directed=False)
        tx_c, ent_c = fcomp[:len(fr)], fcomp[len(fr):]
        used = np.asarray(Bf.sum(0)).ravel() > 0
        anchors_per = np.bincount(ent_c[used & (ent_type == anchor_ti)], minlength=nf)
        for c in np.where(anchors_per >= 2)[0]:
            r_tx = fr[tx_c == c]
            r_ent = np.where(used & (ent_c == c))[0]
            # entities shared by several accounts' frauds are the evidence
            sub_rows = Bf[tx_c == c]
            shared = []
            for e in r_ent[ent_type[r_ent] != anchor_ti]:
                accts = set(tx_ent[r_tx[sub_rows[:, e].nonzero()[0]], anchor_ti].tolist()) - {-1}
                if len(accts) >= 2:
                    shared.append((len(accts), e))
            shared.sort(reverse=True)
            xy = ent_xy[r_ent]
            centre = xy.mean(0)
            rad = float(np.sqrt(((xy - centre) ** 2).sum(1)).max()) + 1.5
            rings.append({"frauds": int(len(r_tx)), "anchors": int(anchors_per[c]), "fraud_anchors": int(anchors_per[c]),
                          "entities": int(len(r_ent)), "transactions": int(len(r_tx)),
                          "x": round(float(centre[0]), 3), "y": round(float(centre[1]), 3), "r": round(rad, 3),
                          "shared": [{"type": types[ent_type[e]], "name": ent_name[e], "accounts": n_a}
                                     for n_a, e in shared[:4]],
                          "tx": r_tx[:500].tolist(), "ents": r_ent[:500].tolist()})
        rings.sort(key=lambda r: (-r["anchors"], -r["frauds"]))

    # ---- pack typed arrays (4-byte arrays first, so every offset stays aligned)
    flags = (fraud_tx.astype(np.uint8) | (np.clip(split, 0, 3).astype(np.uint8) << 1))
    if score is not None and threshold is not None:
        flags |= ((score >= threshold).astype(np.uint8) << 3)
    sc = np.zeros(n, np.uint8) if score is None else np.clip(np.round(score * 255), 0, 255).astype(np.uint8)
    arrays = [("ent_xy", ent_xy.astype(np.float32).ravel()), ("tx_xy", tx_xy.astype(np.float32).ravel()),
              ("tx_ent", tx_ent.ravel()), ("ent_deg", deg), ("ent_fraud", fraud_deg),
              ("ent_type", ent_type), ("tx_flags", flags), ("tx_score", sc)]
    layout, parts, off = {}, [], 0
    for name, arr in arrays:
        b = arr.tobytes()
        layout[name] = {"offset": off, "length": int(arr.size), "dtype": str(arr.dtype)}
        parts.append(b)
        off += len(b)
        pad = (-off) % 4
        if pad:
            parts.append(b"\0" * pad)
            off += pad
    blob = gzip.compress(b"".join(parts), compresslevel=6)

    lo, hi = np.minimum(ent_xy.min(0) if E else [0, 0], tx_xy.min(0)), np.maximum(ent_xy.max(0) if E else [1, 1], tx_xy.max(0))
    sizes = np.bincount(comp[n:][is_struct], minlength=n_comp) if E else np.zeros(1)
    meta = {
        "types": types, "type_labels": [TYPE_LABELS.get(t, t.title()) for t in types],
        "hub_types": hub_types, "anchor_type": types[anchor_ti] if types else None,
        "n_tx": int(n), "n_entities": int(E), "n_edges": int(len(rows)), "n_fraud": int(fraud_tx.sum()),
        "entities_by_type": {t: int((ent_type == i).sum()) for i, t in enumerate(types)},
        "clusters": int((sizes > 0).sum()), "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "multi_anchor_clusters": int(sum(1 for c, mem, _, _ in comp_layout
                                         if (ent_type[mem] == anchor_ti).sum() >= 2)),
        "rings": rings[:30], "n_rings": len(rings), "threshold": threshold, "has_score": score is not None,
        "bounds": [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        "names": ent_name, "arrays": layout,
    }
    return meta, blob


def save_graph(out_dir, meta, blob, tx_info=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph.json.gz").write_bytes(gzip.compress(json.dumps(meta, separators=(",", ":")).encode(), 6))
    (out_dir / "graph.bin.gz").write_bytes(blob)
    if tx_info is not None:
        tx_info.to_pickle(out_dir / "graph_tx.pkl")
