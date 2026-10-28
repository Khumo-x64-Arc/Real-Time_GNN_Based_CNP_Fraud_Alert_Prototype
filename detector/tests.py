"""Tests for the GNN-only pipeline. Run with: python manage.py test detector"""
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from django.test import TestCase

from .ml.engine import LiveEngine
from .ml.gnn import TransactionGNN
from .ml.pipeline import MODEL_NAMES, train_and_evaluate
from .ml.synthgen import generate_unlabelled


def labelled_file(folder: Path, rows=8000) -> Path:
    """Synthetic transactions with the answer key joined on, cut to the period that holds fraud."""
    tx, key = generate_unlabelled(rows=rows, accounts=600, fraud_rate=0.03)
    df = tx.merge(key[["transaction_id", "is_fraud"]], on="transaction_id")
    t = pd.to_datetime(df["timestamp"])
    df = df[t <= t[df["is_fraud"] == 1].max()]
    path = folder / "labelled.csv"
    df.to_csv(path, index=False)
    return path


class GNNPipelineTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        cls.csv = labelled_file(folder)
        cls.out = folder / "model"
        cls.res = train_and_evaluate(cls.csv, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        super().tearDownClass()

    def test_only_gnn_results(self):
        self.assertTrue(set(self.res["models"]) <= set(MODEL_NAMES))
        self.assertIn("GNN", self.res["models"])

    def test_live_test_runs_on_time_split(self):
        if self.res["split"] != "time":
            self.skipTest("this synthetic file fell back to a stratified split")
        self.assertIn("GNN live", self.res["models"])
        self.assertEqual(self.res["live_model"], "GNN live")
        self.assertAlmostEqual(self.res["live_threshold"], self.res["models"]["GNN live"]["threshold"])

    def test_ablation_and_sweep(self):
        abl = self.res["ablation"]
        self.assertEqual(abl[0]["type"], "")
        self.assertEqual({a["type"] for a in abl[1:-1]}, set(self.res["graph_entities"]))
        for a in abl:
            self.assertGreaterEqual(a["pr_auc"], 0)
            self.assertLessEqual(a["pr_auc"], 1)
        self.assertEqual(sum(r["chosen"] for r in self.res["sweep"]["rows"]), 1)

    def test_monitor_reproduces_the_live_test(self):
        """Replaying the test period through the live engine flags exactly what the live test counted."""
        if self.res["split"] != "time":
            self.skipTest("live test needs a time split")
        engine = LiveEngine(self.out / "bundle.joblib")
        replay = pd.read_pickle(self.out / "replay.pkl").to_dict("records")
        flagged = [r["flagged"]["GNN"] for r in engine.score_batch(replay)]
        live = self.res["models"]["GNN live"]
        y = [row["is_fraud"] for row in replay]
        self.assertEqual(sum(f and t for f, t in zip(flagged, y)), live["tp"])
        self.assertEqual(sum(f and not t for f, t in zip(flagged, y)), live["fp"])

    def test_oversampling_comparison_in_results(self):
        """The main run stores its GNN and the counterpart trained the other way, on the same data and split."""
        from .ml.pipeline import prepare, train_variant
        block = self.res["oversampling"]
        self.assertEqual(block["main"], "with")                       # FSMOTE is on by default
        w, wo = block["with"], block["without"]
        self.assertGreater(w["fsmote"]["synthetic"], 0)
        self.assertEqual(wo["fsmote"]["synthetic"], 0)
        self.assertEqual(set(w["models"]), set(wo["models"]))
        self.assertEqual(w["models"]["GNN"]["f1"], self.res["models"]["GNN"]["f1"])   # the main side is the scorecard
        for v in (w, wo):
            for m in v["models"].values():
                self.assertEqual(m["tp"] + m["fn"], self.res["test_frauds"])
        # training is deterministic, so the standalone comparison reproduces the main run's numbers
        P = prepare(self.csv)
        again = train_variant(P, 0.0)
        self.assertEqual(again["models"]["GNN"]["tp"], wo["models"]["GNN"]["tp"])
        self.assertAlmostEqual(again["models"]["GNN"]["pr_auc"], wo["models"]["GNN"]["pr_auc"], places=6)
        json.dumps(block, allow_nan=False)

    def test_explanation_is_gnn_only(self):
        engine = LiveEngine(self.out / "bundle.joblib")
        replay = pd.read_pickle(self.out / "replay.pkl").to_dict("records")
        res = next((r for r in engine.score_batch(replay) if r["level"] != "clear"), None)
        self.assertIsNotNone(res)
        ex = res["explanation"]
        self.assertEqual(ex["model"], "GNN")
        self.assertEqual(set(res["probabilities"]), {"GNN"})
        self.assertTrue(ex["drivers"])


class StreamingConsistencyTests(TestCase):
    def test_score_one_matches_prefix_graph(self):
        """score_one on the history equals the trained GNN run on the past-only graph (bar stale states)."""
        rng = np.random.default_rng(1)
        n, d = 600, 6
        X = rng.normal(size=(n, d))
        acct = [f"a:{i}" for i in rng.integers(0, 40, n)]
        dev = [f"d:{i}" for i in rng.integers(0, 80, n)]
        y = ((X[:, 0] > 1.2) | (np.array([int(a[2:]) for a in acct]) < 3)).astype(int)
        keys = {"account": acct, "device": dev}
        gnn = TransactionGNN(epochs=60).fit(X, keys, y, np.arange(400), np.arange(400, 500))
        live = copy.deepcopy(gnn)
        live.build_inference_state(X[:500], {t: v[:500] for t, v in keys.items()})
        gaps = []
        for i in range(500, 560):
            k = {t: v[i] for t, v in keys.items()}
            exact = gnn.predict_new(X[: i + 1], {t: v[: i + 1] for t, v in keys.items()})[-1]
            gaps.append(abs(live.score_one(X[i], k, update=True) - exact))
        self.assertLess(float(np.mean(gaps)), 0.03)


class RLAgentTests(TestCase):
    """Semi-supervised GNN + DQN on small synthetic look-alikes of the two benchmark files."""

    def test_ulb_pipeline_end_to_end(self):
        from .ml.rl_data import synth_ulb
        from .ml.rl_pipeline import POLICY_ORDER, run_experiment
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "creditcard.csv"
            synth_ulb(n=20_000, fraud_rate=0.02).to_csv(path, index=False)
            res = run_experiment("ulb", path, Path(d) / "out", label_rate=0.3, rounds=1, episodes=25, epochs=40,
                                 knn_min_sim=0.5, full_reference=False, agent_seeds=1)
        self.assertEqual(set(res["policies"]), set(POLICY_ORDER))
        for p in res["policies"].values():
            self.assertLessEqual(p["tp"] + p["fn"], res["dataset"]["test_frauds"] + 1e-9)
            self.assertGreaterEqual(p["pr_auc"], 0)
        self.assertEqual(len(res["training"]["agent"]), 25)
        self.assertIn("semi", res["semi_supervised"]["report"])

    def test_ieee_loader_and_causal_streaming(self):
        from .ml.rl_data import load_ieee, synth_ieee
        from .ml.semisup import causal_scores
        with tempfile.TemporaryDirectory() as d:
            tr, ident = synth_ieee(n=6000, days=20)
            tr.to_csv(Path(d) / "train_transaction.csv", index=False)
            ident.to_csv(Path(d) / "train_identity.csv", index=False)
            g = load_ieee(d)
        self.assertEqual(set(g.entity_keys), {"account", "card", "device"})
        self.assertEqual(g.read.shape, g.write.shape)
        a, b, c = g.split()
        gnn = TransactionGNN(epochs=20).fit(g.X, g.entity_keys, g.y, a, b)
        z, H = causal_scores(gnn, g, int(b[0]))
        self.assertEqual(len(z), g.n - int(b[0]))
        self.assertEqual(H.shape[1], gnn.hidden)

    def test_similarity_graph_is_causal(self):
        from .ml.rl_data import causal_knn
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 20_000, 3000))
        A, _ = causal_knn(rng.normal(size=(3000, 8)), t, k=4, min_sim=0.2)
        r, c = A.nonzero()
        self.assertTrue((c < r).all())
        self.assertTrue((t[r] - t[c] <= 3600).all())

    def test_branching_agent_learns_a_bandit(self):
        """On a toy stream where only a high threshold pays, the agent should learn to choose it."""
        import scipy.sparse as sp
        from .ml import rl_agent as R
        rng = np.random.default_rng(1)
        n = 6000
        y = (rng.uniform(size=n) < 0.05).astype(int)
        z = np.where(y == 1, rng.normal(2.0, 1, n), rng.normal(-1.0, 1.2, n))
        H = rng.normal(size=(n, 8)).astype(np.float32)
        I = sp.identity(n, format="csr")
        wins = R.windows_for(np.arange(n, dtype=float), np.arange(n), 200)
        env = R.FraudStreamEnv(z, H, y, I, I, wins, prior=0.05, theta0=-1.0, label_rate=1.0, delay=0,
                               projector=R.make_projector(H))
        agent, _ = R.train_agent(env, episodes=120, seed=3)
        e = R.run_policy(env, "agent", np.ones(n, bool), agent=agent)
        offsets = [R.ACTIONS[t["action"]][0] for t in e.trace]
        self.assertGreater(np.mean(offsets), 0.5)   # it raised the threshold well above the bad starting point


def sa_target_file(folder: Path, rows=3000, seed=0) -> Path:
    """A small file in the synthetic South African schema: fraud is e-commerce, larger, on new devices, often abroad."""
    rng = np.random.default_rng(seed)
    acc = rng.integers(0, 200, rows)
    t = np.sort(rng.uniform(0, 20 * 86400, rows))
    fraud = rng.random(rows) < 0.03
    amount = np.where(fraud, rng.lognormal(6, .8, rows), rng.lognormal(3, 1, rows)).round(2)
    df = pd.DataFrame({
        "transaction_id": [f"TX{i:06d}" for i in range(rows)], "Time": t, "Amount": amount,
        "account_id": [f"A{a}" for a in acc], "card_token": [f"C{a}" for a in acc],
        "device_id": [f"NEW{i}" if f else f"D{a}" for i, (a, f) in enumerate(zip(acc, fraud))],
        "merchant_id": [f"M{m}" for m in rng.integers(0, 80, rows)],
        "tx_country": np.where(fraud & (rng.random(rows) < .4), "NG", "ZA"),
        "account_balance": rng.uniform(500, 20000, rows).round(2),
        "channel": np.where(fraud, "ECOM", rng.choice(["POS", "ECOM"], rows)),
        "entry_mode": np.where(fraud, "ecom_no_3ds", "chip"), "merchant_category": "grocery",
        "device_type": "android", "card_product": "classic", "account_tenure_months": 24,
        "seconds_since_prev_account_tx": 3600.0, "Class": fraud.astype(int)})
    path = folder / "sa_target.csv"
    df.to_csv(path, index=False)
    return path


class AutomationTests(TestCase):
    def test_cases_link_bursts_devices_and_cash_out_merchants(self):
        from .ml.fraud_automation import group_cases
        h = 3600
        rows = pd.DataFrame([
            # account A takeover burst
            {"i": 0, "t": 0, "account": "A", "card": "cA", "device": "d1", "ip": None, "merchant": "m1"},
            {"i": 1, "t": 600, "account": "A", "card": "cA", "device": "d1", "ip": None, "merchant": "m2"},
            # account B on the same device: a ring
            {"i": 2, "t": 2 * h, "account": "B", "card": "cB", "device": "d1", "ip": None, "merchant": "m3"},
            # accounts C and D cash out at the same merchant within 72 h: a ring
            {"i": 3, "t": 10 * h, "account": "C", "card": "cC", "device": "d3", "ip": None, "merchant": "mx"},
            {"i": 4, "t": 40 * h, "account": "D", "card": "cD", "device": "d4", "ip": None, "merchant": "mx"},
            # E at the same merchant but much later: not linked
            {"i": 5, "t": 200 * h, "account": "E", "card": "cE", "device": "d5", "ip": None, "merchant": "mx"},
            # A again 60 h later on a new device: outside the 24 h account window, but the same card links it
            {"i": 6, "t": 60 * h, "account": "A", "card": "cA", "device": "d9", "ip": None, "merchant": "m7"},
        ])
        # two unrelated escalations with no device or IP (NaN, as pandas stores it) must not be linked
        rows = pd.concat([rows, pd.DataFrame([
            {"i": 7, "t": 300 * h, "account": "F", "card": "cF", "device": np.nan, "ip": np.nan, "merchant": "mf"},
            {"i": 8, "t": 301 * h, "account": "G", "card": "cG", "device": np.nan, "ip": np.nan, "merchant": "mg"}])],
            ignore_index=True)
        root, links = group_cases(rows)
        self.assertNotEqual(root[7], root[8])
        self.assertEqual(root[0], root[1])
        self.assertEqual(root[0], root[2])           # shared device across accounts
        self.assertEqual(root[3], root[4])           # shared cash-out merchant within 72 h
        self.assertNotEqual(root[3], root[0])
        self.assertNotEqual(root[5], root[4])        # too late to be the same campaign
        self.assertEqual(root[6], root[0])           # same card
        kinds = {k for _, _, k in links[root[0]]}
        self.assertTrue({"account", "device", "card"} <= kinds)

    def test_automation_end_to_end(self):
        from .ml.fraud_automation import run_automation
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = sa_target_file(d, rows=4000)
            res = run_automation(path, {}, d / "out", episodes=3, agent_seeds=1, window_hours=12, delay=1)
            a = res["automation"]
            self.assertEqual(set(res["bases"]), {"target_only"})
            self.assertEqual(sum(f["n"] for f in a["funnel"]), res["target"]["test_rows"])
            dec = pd.read_csv(d / "out" / "decisions.csv")
            self.assertEqual(len(dec), res["target"]["test_rows"])
            # every escalation is explained, and so is every held-back transaction
            self.assertTrue(dec.loc[dec["decision"].isin(["escalate", "block_case", "held_back"]), "reason"].notna().all())
            # escalated = score at or above the agent's threshold for that window
            esc = dec["decision"].isin(["escalate", "block_case"])
            self.assertTrue((dec.loc[esc, "decision_score"] >= dec.loc[esc, "threshold"] - 1e-6).all())
            self.assertTrue((dec.loc[~esc, "decision_score"] < dec.loc[~esc, "threshold"] + 1e-6).all())
            self.assertTrue((d / "out" / "automation.joblib").exists())
            json.dumps(res, allow_nan=False)                 # stores as JSON (no NaN)


def sa_full_file(folder: Path, rows=5000, accounts=160, seed=0) -> Path:
    """A small file with every column of the synthetic South African schema that ring injection reads."""
    rng = np.random.default_rng(seed)
    acc = rng.integers(0, accounts, rows)
    t = np.sort(rng.uniform(0, 24 * 86400, rows)).round()
    fraud = rng.random(rows) < 0.02
    amount = np.where(fraud, rng.lognormal(6, .8, rows), rng.lognormal(3, .8, rows)).round(2)
    cats = np.array(["grocery", "fuel", "electronics", "gift_cards", "online_marketplace", "streaming", "crypto_exchange"])
    m = rng.integers(0, 70, rows)
    df = pd.DataFrame({
        "transaction_id": [f"TX{i:08d}" for i in range(rows)], "Time": t, "Amount": amount,
        "account_id": [f"ACC{a:05d}" for a in acc], "account_number": 6200000000 + acc, "cardholder_name": [f"N{a}" for a in acc],
        "customer_ref": [f"CR{a}" for a in acc], "issuing_bank": "Bank", "account_tenure_months": 24,
        "home_city": "Johannesburg", "home_province": "Gauteng", "card_token": [f"C{a}" for a in acc],
        "card_scheme": "VISA", "card_product": "classic", "card_last4": 1000 + acc,
        "merchant_id": [f"MER{x:06d}" for x in m], "merchant_name": [f"Merchant {x}" for x in m],
        "merchant_category": cats[m % len(cats)], "channel": np.where(fraud | (m % 2 == 0), "ECOM", "POS"),
        "entry_mode": np.where(fraud, "ecom_no_3ds", "chip"), "terminal_id": "",
        "device_id": [f"NEW{i}" if f else f"D{a}" for i, (a, f) in enumerate(zip(acc, fraud))],
        "device_type": "android", "ip_address": [f"41.1.{a % 4}.{a % 250 + 2}" for a in acc],
        "tx_country": np.where(fraud & (rng.random(rows) < .4), "NG", "ZA"), "tx_city": "Johannesburg",
        "Class": fraud.astype(float)})
    ts = pd.Timestamp("2013-09-02") + pd.to_timedelta(df["Time"], unit="s")
    df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["transaction_date"] = ts.dt.strftime("%Y-%m-%d")
    df["posting_date"] = (ts.dt.normalize() + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df["reference"] = np.arange(rows) + 10**9
    df = df.sort_values(["Time", "transaction_id"]).reset_index(drop=True)
    g = df.groupby("account_id")
    df["account_tx_seq"] = g.cumcount() + 1
    df["seconds_since_prev_account_tx"] = g["Time"].diff()
    df["account_balance"] = (60000 - g["Amount"].cumsum()).round(2)
    path = folder / "sa_full.csv"
    df.to_csv(path, index=False)
    return path


class FraudRingTests(TestCase):
    def test_injected_rings_keep_the_file_consistent(self):
        from .ml.fraud_rings import RING_COLUMNS, inject_rings
        with tempfile.TemporaryDirectory() as d:
            src = pd.read_csv(sa_full_file(Path(d)), low_memory=False)
        out, summary = inject_rings(src, seed=5)
        self.assertEqual(list(summary.index), ["R01", "R02", "R03", "R04", "R05", "R06"])
        self.assertEqual(list(summary["period"]), ["train", "train", "train", "validation", "test", "test"])
        self.assertGreater(len(out), len(src))
        self.assertTrue(out["transaction_id"].is_unique)
        self.assertTrue(out["Time"].is_monotonic_increasing)
        self.assertGreaterEqual(out["account_balance"].min(), 0)
        # original rows keep every value except the derived per-account fields
        m = src.merge(out, on="transaction_id", suffixes=("_o", "_n"))
        self.assertEqual(len(m), len(src))
        for c in src.columns:
            if c in ("transaction_id", "account_tx_seq", "seconds_since_prev_account_tx", "account_balance"):
                continue
            a, b = m[c + "_o"], m[c + "_n"]
            self.assertTrue((a.eq(b) | (a.isna() & b.isna())).all(), c)
        # derived fields agree with the transactions, account by account
        for _, g in out.groupby("account_id"):
            self.assertEqual(list(g["account_tx_seq"]), list(range(1, len(g) + 1)))
            self.assertTrue(np.allclose(np.diff(g["Time"]), g["seconds_since_prev_account_tx"].iloc[1:]))
            self.assertTrue(np.allclose(np.diff(g["account_balance"]), -g["Amount"].iloc[1:], atol=.011))
        rid = out["fraud_ring_id"].fillna("")
        self.assertTrue((out.loc[rid != "", "Class"] == 1).all())
        self.assertEqual(set(RING_COLUMNS) - set(out.columns), set())
        # injecting twice is refused
        from .ml.fraud_rings import RingError
        with self.assertRaises(RingError):
            inject_rings(out)

    def test_ring_labels_are_never_features(self):
        from .ml import transfer_rl as T
        from .ml.fraud_rings import inject_rings
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out, _ = inject_rings(pd.read_csv(sa_full_file(d), low_memory=False), seed=5)
            out.to_csv(d / "a.csv", index=False)
            scrambled = out.assign(fraud_ring_id=np.where(out["fraud_ring_id"].fillna("") != "", "X", ""),
                                   fraud_pattern="zzz")
            scrambled.to_csv(d / "b.csv", index=False)
            Xa = T.features(T.load_sa(d / "a.csv")[0])
            Xb = T.features(T.load_sa(d / "b.csv")[0])
        self.assertTrue(np.allclose(np.asarray(Xa, float), np.asarray(Xb, float), equal_nan=True))

    def test_score_known_rings(self):
        from .ml.fraud_rings import score_known_rings
        h = 3600
        dec = pd.DataFrame({
            "transaction_id": [f"T{i}" for i in range(7)], "t": [h * i for i in range(7)],
            "timestamp": [str(i) for i in range(7)],
            "account": ["a1", "a2", "a3", "b1", "b2", "b3", "x"], "device": ["d", "d", "d", "e1", "e2", "e3", "z"],
            "ip": [None] * 7, "merchant": ["m", "m", "m", "n1", "n2", "n3", "m"], "merchant_name": ["M"] * 7,
            "amount": [100.0] * 7, "gnn_logit": [3, 3, -1, -5, -5, 2, 0], "gnn_prob": [.9] * 7,
            "threshold": [0.0] * 7, "threshold_offset": [0.0] * 7,
            "escalated": [True, True, True, False, False, True, True],
            "decision": ["block_case", "block_case", "block_case", "approve", "held_back", "escalate", "block_case"],
            "case": ["C1", "C1", "C1", None, None, None, "C1"], "case_type": ["ring"] * 3 + [None] * 3 + ["ring"],
            "ring_linked": [None] * 7, "is_fraud": [1, 1, 1, 1, 1, 1, 0],
            "known_ring": ["R5", "R5", "R5", "R6", "R6", "R6", ""]})
        rows = pd.DataFrame({"ring": ["R1", "R1", "R5", "R5", "R5", "R6", "R6", "R6"],
                             "pattern": ["card_testing"] * 8, "period": ["train"] * 2 + ["test"] * 6,
                             "account": ["p", "q", "a1", "a2", "a3", "b1", "b2", "b3"],
                             "revealed": [True, False] + [False] * 6, "is_fraud": [1] * 8})
        res = {r["ring"]: r for r in score_known_rings(dec, rows)}
        self.assertEqual(res["R1"]["revealed_labels"], 1)
        self.assertNotIn("scored", res["R1"])
        r5, r6 = res["R5"], res["R6"]
        self.assertEqual((r5["in_ring_case"], r5["escalated"], r5["gnn_alone"]), (3, 3, 2))
        self.assertEqual(r5["outcome"], "caught as one ring case")
        self.assertEqual(r5["cases"][0]["legit"], 1)
        self.assertEqual((r6["in_ring_case"], r6["escalated"], r6["held_back"], r6["approved"]), (0, 1, 1, 1))
        self.assertEqual(r6["outcome"], "escalated, but not linked as a ring")
        self.assertTrue(r5["network"]["nodes"] and r5["network"]["edges"])

    def test_automation_scores_planted_rings_and_page_shows_them(self):
        from unittest import mock

        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.urls import reverse

        from django.contrib.auth.models import User

        from .ml.fraud_automation import run_automation
        from .models import AutomationRun
        self.client.force_login(User.objects.create_user("analyst", password="x", is_staff=True))
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            d = Path(d)
            raw = sa_full_file(d).read_bytes()
            # the upload form plants the rings and starts a run
            with mock.patch("detector.automation_views.start_automation") as start:
                r = self.client.post(reverse("automation"), {
                    "file": SimpleUploadedFile("sa.csv", raw, content_type="text/csv"), "name": "rings",
                    "inject_rings": "on", "ring_seed": "5"})
            self.assertEqual(r.status_code, 302)
            run = AutomationRun.objects.get(name="rings")
            start.assert_called_once_with(run.pk)
            self.assertEqual([x["ring"] for x in run.params["rings"]], ["R01", "R02", "R03", "R04", "R05", "R06"])
            self.assertTrue(run.file.name.endswith("_rings.csv"))
            res = run_automation(run.file.path, {}, run.run_dir, episodes=2, agent_seeds=1, window_hours=12, delay=1)
            self.assertEqual(set(res["bases"]), {"target_only"})
            kr = {k["ring"]: k for k in res["automation"]["known_rings"]}
            self.assertEqual(set(kr), {"R01", "R02", "R03", "R04", "R05", "R06"})
            for k in ("R05", "R06"):
                s = kr[k]
                self.assertTrue(s["scored"])
                self.assertEqual(s["escalated"], s["in_ring_case"] + s["in_burst"] + s["escalated_alone"])
                self.assertEqual(s["transactions"], s["escalated"] + s["held_back"] + s["approved"])
            dec = pd.read_csv(run.run_dir / "decisions.csv", low_memory=False)
            self.assertEqual(int((dec["known_ring"].fillna("").isin(["R05", "R06"]) & (dec["is_fraud"] == 1)).sum()),
                             kr["R05"]["transactions"] + kr["R06"]["transactions"])
            json.dumps(res, allow_nan=False)
            AutomationRun.objects.filter(pk=run.pk).update(status="ready", results=res)
            page = self.client.get(run.get_absolute_url())
            self.assertContains(page, "Planted fraud rings")
            self.assertContains(page, "R05")
            self.assertContains(page, "<svg viewBox", html=False)
            self.assertEqual(self.client.get(reverse("automation_file", args=[run.pk])).status_code, 200)
            csv = self.client.get(reverse("automation_decisions", args=[run.pk]) + "?decision=R06")
            self.assertEqual(csv.status_code, 200)
            self.assertContains(self.client.get(reverse("automation")), "Planted ring tx in ring cases")


class AdversarialTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = pd.read_csv(sa_full_file(Path(cls.tmp.name), rows=6000, accounts=220), low_memory=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        super().tearDownClass()

    def test_attack_groups_keep_the_file_consistent_and_edit_what_they_claim(self):
        from .ml.adversarial import AdversarialError, GROUPS, inject_adversarial
        out = inject_adversarial(self.src, victims_per_group=4, optimised_edits=("country",))
        a = out[out["adversarial_group"].fillna("") != ""]
        self.assertEqual(set(a["adversarial_group"]), set(GROUPS))
        self.assertTrue((a["Class"] == 1).all())
        self.assertEqual(set(a["adversarial_period"]), {"test"})
        self.assertTrue(out["transaction_id"].is_unique)
        self.assertGreaterEqual(out["account_balance"].min(), 0)
        m = self.src.merge(out, on="transaction_id", suffixes=("_o", "_n"))
        for c in self.src.columns:
            if c in ("transaction_id", "account_tx_seq", "seconds_since_prev_account_tx", "account_balance"):
                continue
            x, y = m[c + "_o"], m[c + "_n"]
            self.assertTrue((x.eq(y) | (x.isna() & y.isna())).all(), c)
        for _, g in out.groupby("account_id"):
            self.assertEqual(list(g["account_tx_seq"]), list(range(1, len(g) + 1)))
            self.assertTrue(np.allclose(np.diff(g["account_balance"]), -g["Amount"].iloc[1:], atol=.011))
        # full mimicry: the victim's own device, home country, one transaction a day
        for acc, g in a[a["adversarial_group"] == "A1"].groupby("account_id"):
            hist = self.src[self.src["account_id"] == acc]
            self.assertTrue(set(g["device_id"]) <= set(hist["device_id"].dropna()))
            self.assertTrue((g["tx_country"] == hist["tx_country"].mode().iloc[0]).all())
            self.assertEqual(g["transaction_date"].nunique(), len(g))
        # the optimised group with the home-country edit stays at home; the control group mostly does not
        self.assertTrue((a.loc[a["adversarial_group"] == "A4", "tx_country"] == "ZA").all())
        cam = a[a["adversarial_group"] == "A3"]
        self.assertEqual(cam["device_id"].nunique(), 1)          # one busy legitimate hub device
        with self.assertRaises(AdversarialError):
            inject_adversarial(out, victims_per_group=4)

    def test_adversarial_labels_are_never_features(self):
        from .ml import transfer_rl as T
        from .ml.adversarial import inject_adversarial
        out = inject_adversarial(self.src, victims_per_group=4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out.to_csv(d / "a.csv", index=False)
            out.assign(adversarial_group=np.where(out["adversarial_group"] != "", "Z9", ""),
                       adversarial_attack="x", adversarial_period="y").to_csv(d / "b.csv", index=False)
            Xa = T.features(T.load_sa(d / "a.csv")[0])
            Xb = T.features(T.load_sa(d / "b.csv")[0])
        self.assertTrue(np.allclose(np.asarray(Xa, float), np.asarray(Xb, float), equal_nan=True))

    def test_white_box_attack_and_adversarial_training_path(self):
        from .ml.adversarial import build_adversarial_file
        out, info = build_adversarial_file(self.src, victims_per_group=4, adversarial_training=True,
                                           progress=lambda m: None)
        self.assertTrue(info["adversarial_training"])
        self.assertEqual(len(info["paths"]), 2)
        for p in info["paths"]:
            flagged = [s["flagged"] for s in p["steps"]]
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(flagged, flagged[1:])))   # each edit kept helps
        self.assertEqual(set(out.loc[out["adversarial_group"] != "", "adversarial_period"]), {"train", "test"})
        json.dumps(info, allow_nan=False)

    def test_score_groups(self):
        from .ml.adversarial import adversarial_summary
        dec = pd.DataFrame({"adversarial_group": ["C0", "C0", "A1", "A1", "", "", ""],
                            "is_fraud": [1, 1, 1, 1, 1, 0, 0], "account": list("aabbcde"),
                            "gnn_logit": [2.0, -3, -5, -5, 1, -2, 3], "gnn_prob": [.9, .1, 0, 0, .7, .1, .9],
                            "escalated": [True, True, False, False, True, False, True],
                            "case": ["C1", "C1", None, None, None, None, None],
                            "decision": ["block_case", "block_case", "approve", "held_back", "escalate", "approve", "escalate"],
                            "amount": [100.0, 200, 10, 20, 50, 5, 5]})
        s = adversarial_summary(dec, 0.0, [1, -1, 0, 0])
        g = {x["group"]: x for x in s["groups"]}
        self.assertEqual((g["C0"]["gnn_alone"], g["C0"]["escalated"], g["C0"]["value_escalated"]), (1, 2, 300.0))
        self.assertEqual((g["A1"]["escalated"], g["A1"]["held_back"], g["A1"]["approved"]), (0, 1, 1))
        self.assertEqual(s["native_frauds"], {"n": 1, "gnn_alone": 1, "escalated": 1})
        self.assertEqual(s["false_alarms"], 1)
        self.assertAlmostEqual(s["windows_threshold_lowered"], .25)

    def test_form_run_and_page(self):
        from unittest import mock

        from django.contrib.auth.models import User
        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.urls import reverse

        from . import tasks
        from .ml.fraud_automation import run_automation
        from .models import AutomationRun
        self.client.force_login(User.objects.create_user("analyst2", password="x", is_staff=True))
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            buf = self.src.to_csv(index=False).encode()
            with mock.patch("detector.automation_views.start_automation"):
                r = self.client.post(reverse("automation"), {
                    "file": SimpleUploadedFile("sa.csv", buf, content_type="text/csv"), "name": "adv",
                    "inject_adversarial": "on"})
            self.assertEqual(r.status_code, 302)
            run = AutomationRun.objects.get(name="adv")
            self.assertEqual(run.params["adversarial"], {"training": False, "seed": 7})
            tasks._plant_adversarial(run, run.params["adversarial"], lambda m, p: None)
            run.refresh_from_db()
            self.assertTrue(run.file.name.endswith("_adversarial.csv"))
            setup = run.params["adversarial_setup"]
            res = run_automation(run.file.path, {}, run.run_dir, episodes=2, agent_seeds=1, window_hours=12, delay=1)
            res["adversarial_setup"] = setup
            adv = res["automation"]["adversarial"]
            self.assertEqual({g["group"] for g in adv["groups"]} - {"C0", "A1", "A2", "A3", "A4"}, set())
            for g in adv["groups"]:
                self.assertEqual(g["transactions"], g["escalated"] + g["held_back"] + g["approved"])
            json.dumps(res, allow_nan=False)
            AutomationRun.objects.filter(pk=run.pk).update(status="ready", results=res)
            page = self.client.get(run.get_absolute_url())
            self.assertContains(page, "Adversarial examples: can the GNN and the agent detect them?")
            self.assertContains(page, "How the white-box attack evaded the GNN")
            csv = self.client.get(reverse("automation_decisions", args=[run.pk]) + "?decision=A4")
            self.assertEqual(csv.status_code, 200)
            self.assertContains(self.client.get(reverse("automation")), "Adversarial frauds escalated")


class TransferPagesRemovedTests(TestCase):
    def test_no_transfer_pages_or_nav_link(self):
        from django.contrib.auth.models import User
        from django.urls import NoReverseMatch, reverse
        for name in ("transfer", "finetune", "transfer_run", "transfer_model"):
            with self.assertRaises(NoReverseMatch):
                reverse(name)
        self.client.force_login(User.objects.create_user("analyst3", password="x", is_staff=True))
        self.assertEqual(self.client.get("/transfer/").status_code, 404)
        page = self.client.get(reverse("automation"))
        self.assertEqual(page.status_code, 200)
        self.assertNotContains(page, ">Transfer<")
        self.assertNotContains(page, "source GNNs")


class OversamplingPageTests(TestCase):
    def test_results_page_section_button_and_export(self):
        from unittest import mock

        from django.contrib.auth.models import User
        from django.urls import reverse

        from .models import Dataset, ModelResult
        self.client.force_login(User.objects.create_user("analyst4", password="x", is_staff=True))
        m = {"roc_auc": .95, "pr_auc": .8, "f1": .7, "accuracy": .99, "precision": .75, "recall": .66,
             "threshold": .5, "tp": 66, "fp": 22, "tn": 9000, "fn": 34, "train_seconds": 3.0}
        res = {"version": "gnn-only-1", "rows": 10000, "fraud_rate": .01, "test_frauds": 100, "split": "time",
               "sizes": {"test": 9122}, "models": {"GNN": m, "GNN live": m}, "fsmote": {"synthetic": 50}}
        ds = Dataset.objects.create(name="ds", status=Dataset.Status.READY, results=res)
        for name in ("GNN", "GNN live"):
            ModelResult.objects.create(dataset=ds, name=name, **m)
        page = self.client.get(reverse("results") + f"?dataset={ds.pk}")
        self.assertContains(page, "With and without oversampling")
        self.assertContains(page, "Compare with and without oversampling")          # older dataset: offer the run
        self.assertContains(page, "In the study: the semi-supervised GNN and the RL agent")
        with mock.patch("detector.tasks.start_oversampling_comparison") as start:
            r = self.client.post(reverse("dataset_oversampling", args=[ds.pk]))
        self.assertEqual(r.status_code, 302)
        start.assert_called_once_with(ds.pk)
        better = {**m, "pr_auc": .85, "fp": 11}
        ds.results = {**res, "oversampling": {"main": "with", "computed_at": "now",
                                             "with": {"models": {"GNN": m, "GNN live": m}, "fsmote": {"synthetic": 50},
                                                      "train_seconds": 3, "best_val_ap": .8},
                                             "without": {"models": {"GNN": better, "GNN live": better},
                                                         "fsmote": {"synthetic": 0}, "train_seconds": 2,
                                                         "best_val_ap": .82}}}
        ds.save()
        page = self.client.get(reverse("results") + f"?dataset={ds.pk}")
        self.assertContains(page, "With FSMOTE (50 synthetic frauds)")
        self.assertContains(page, "+0.050")                     # PR-AUC change
        self.assertContains(page, "\u221211")                    # 11 fewer false alarms
        csv = self.client.get(reverse("results_export")).content.decode()
        self.assertIn("GNN · live (streaming) (without oversampling)", csv)
