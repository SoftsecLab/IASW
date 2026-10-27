#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute gated vs fisher-only TPR on dolly/wiki llama3.1-8b-instruct.

Reads main_exp dual_kgw_iasw.jsonl, extracts:
  - Before Attack: detection_wm.z_score_micro / z_score_fisher
  - Each attack:   attacks[<attack>].detection_results.z_score_micro / z_score_fisher
  - No-watermark calibration: detection_no_wm.z_score_micro / z_score_fisher

Calibrates tau_KGW = 2.0 fixed, then:
  - tau_F_orig : lowest Fisher threshold such that no_wm FPR <= 1% (fisher-only)
  - tau_F_gated: lowest Fisher threshold such that
                 mean((z_KGW >= tau_KGW) & (z_Fisher >= tau_F_gated)) <= 1% on no_wm

Pure CPU, no GPU needed.
"""
import json
import os
import csv
import numpy as np

ROOT = r"D:\桌面\小论文2组会\IASW"
DATASETS = ["dolly", "wiki"]
MODEL = "llama3.1-8b-instruct"
KGW_GATE = 2.0
TARGET_FPR = 0.01
# attacks to include (paper protocol)
ATTACKS = ["typo", "drop", "swap", "del_ins", "pegasus",
           "dipper", "back_translation", "copy_paste"]


def load_scores(path):
    before_zk, before_zf = [], []
    atk_zk = {a: [] for a in ATTACKS}
    atk_zf = {a: [] for a in ATTACKS}
    no_zk, no_zf = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Before attack
            dw = r.get("detection_wm") or {}
            if dw.get("z_score_micro") is not None:
                before_zk.append(float(dw["z_score_micro"]))
                before_zf.append(float(dw["z_score_fisher"]))

            # No-wm calibration
            dn = r.get("detection_no_wm") or {}
            if dn.get("z_score_micro") is not None:
                no_zk.append(float(dn["z_score_micro"]))
                no_zf.append(float(dn["z_score_fisher"]))

            # Attacks
            for a in ATTACKS:
                node = (r.get("attacks") or {}).get(a) or {}
                dr = node.get("detection_results") or {}
                if dr.get("z_score_micro") is not None:
                    atk_zk[a].append(float(dr["z_score_micro"]))
                    atk_zf[a].append(float(dr["z_score_fisher"]))

    return {
        "before_zk": np.asarray(before_zk),
        "before_zf": np.asarray(before_zf),
        "atk_zk": {a: np.asarray(atk_zk[a]) for a in ATTACKS},
        "atk_zf": {a: np.asarray(atk_zf[a]) for a in ATTACKS},
        "no_zk": np.asarray(no_zk),
        "no_zf": np.asarray(no_zf),
    }


def calibrate_fisher_only(zf_no, target_fpr):
    # lowest threshold s.t. mean(zf >= tau) <= target_fpr
    order = np.argsort(zf_no)
    zf_sorted = zf_no[order]
    n = len(zf_no)
    for i in range(n):
        thr = zf_sorted[i]
        fpr = float(np.mean(zf_no >= thr))
        if fpr <= target_fpr:
            return float(thr), fpr
    return float(zf_no.max() + 1e-6), 0.0


def calibrate_gated(zk_no, zf_no, kgw_gate, target_fpr):
    gate_mask = zk_no >= kgw_gate
    n = len(zk_no)
    candidates = np.sort(zf_no[gate_mask])
    for thr in candidates:
        acc = float(np.mean(gate_mask & (zf_no >= thr)))
        if acc <= target_fpr:
            return float(thr), acc
    return float(zf_no[gate_mask].max() + 1e-6), 0.0


def tpr_fisher(zf, thr):
    return float(np.mean(zf >= thr)) if len(zf) else float("nan")


def tpr_gated(zk, zf, kgw_gate, thr):
    return float(np.mean((zk >= kgw_gate) & (zf >= thr))) if len(zk) else float("nan")


out_rows = []
for ds in DATASETS:
    p = os.path.join(ROOT, "results/main_exp", ds, MODEL, "records",
                     "dual_kgw_iasw.jsonl")
    if not os.path.exists(p):
        print(f"[skip] {p}")
        continue
    print(f"\n=== {ds} / {MODEL} ===")
    d = load_scores(p)
    print(f"  n_before={len(d['before_zk'])}, n_no={len(d['no_zk'])}")

    tau_orig, fpr_orig = calibrate_fisher_only(d["no_zf"], TARGET_FPR)
    tau_gated, fpr_gated = calibrate_gated(d["no_zk"], d["no_zf"],
                                           KGW_GATE, TARGET_FPR)
    print(f"  tau_F_orig={tau_orig:.4f} (empirical FPR={fpr_orig:.4f})")
    print(f"  tau_F_gated={tau_gated:.4f} (empirical FPR={fpr_gated:.4f})")

    # before
    fo = tpr_fisher(d["before_zf"], tau_orig)
    ga = tpr_gated(d["before_zk"], d["before_zf"], KGW_GATE, tau_gated)
    print(f"  Before: fisher={fo*100:.1f}% gated={ga*100:.1f}% d={(ga-fo)*100:+.1f}pp")
    out_rows.append({"dataset": ds, "condition": "Before Attack",
                     "n": len(d["before_zk"]),
                     "fisher_only": round(fo*100, 1),
                     "gated": round(ga*100, 1),
                     "delta_pp": round((ga-fo)*100, 1)})

    for a in ATTACKS:
        n = len(d["atk_zk"][a])
        fo = tpr_fisher(d["atk_zf"][a], tau_orig)
        ga = tpr_gated(d["atk_zk"][a], d["atk_zf"][a], KGW_GATE, tau_gated)
        print(f"  {a:<20} n={n} fisher={fo*100:6.1f}% gated={ga*100:6.1f}% d={(ga-fo)*100:+5.1f}pp")
        out_rows.append({"dataset": ds, "condition": a, "n": n,
                         "fisher_only": round(fo*100, 1),
                         "gated": round(ga*100, 1),
                         "delta_pp": round((ga-fo)*100, 1)})

out_csv = os.path.join(ROOT, "results/security_validation/gate_effect_dolly_wiki.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "condition", "n",
                                     "fisher_only", "gated", "delta_pp"])
    w.writeheader()
    w.writerows(out_rows)
print(f"\nSaved: {out_csv}")
