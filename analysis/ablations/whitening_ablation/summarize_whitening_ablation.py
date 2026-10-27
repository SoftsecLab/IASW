#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize whitening ablation JSONL files; outputs metrics and paired on-minus-off deltas."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, roc_curve

CHANNELS=("z_score","z_score_micro","z_score_macro","z_score_max","z_score_fisher")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--white_dir",required=True)
    p.add_argument("--no_white_dir",required=True)
    p.add_argument("--output_csv",required=True)
    p.add_argument("--negative_source",choices=("human","no_wm"),default="human")
    return p.parse_args()

def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except Exception as e: raise RuntimeError(f"Invalid JSON {path}:{n}: {e}") from e
    return rows

def score(obj,ch):
    if not isinstance(obj,dict): return None
    candidates=[obj]
    for k in ("details","result","macro","micro"):
        if isinstance(obj.get(k),dict): candidates.append(obj[k])
    for d in candidates:
        v=d.get(ch)
        if isinstance(v,(int,float)) and math.isfinite(float(v)): return float(v)
    return None

def metrics(neg,pos):
    neg=np.asarray([x for x in neg if np.isfinite(x)],dtype=float)
    pos=np.asarray([x for x in pos if np.isfinite(x)],dtype=float)
    if not len(neg) or not len(pos): return None
    y=np.r_[np.zeros(len(neg),dtype=int),np.ones(len(pos),dtype=int)]
    s=np.r_[neg,pos]
    fpr,tpr,_=roc_curve(y,s)
    def at(target):
        ids=np.flatnonzero(fpr<=target)
        return float(tpr[ids[-1]]) if len(ids) else 0.0
    return {"AUROC":float(auc(fpr,tpr)),"TPR@1%FPR":at(.01),"TPR@5%FPR":at(.05),
            "n_neg":int(len(neg)),"n_pos":int(len(pos))}

def discover_files(folder, suffix):
    found={}
    for p in Path(folder).glob(f"*_{suffix}.jsonl"):
        method=p.name[:-len(f"_{suffix}.jsonl")]
        found[method]=p
    return found

def summarize_file(path,method,condition,negative_source):
    rows=read_jsonl(path)
    out={}
    for ch in CHANNELS:
        neg=[]
        for r in rows:
            obj=r.get("detection_human" if negative_source=="human" else "detection_no_wm",{})
            v=score(obj,ch)
            if v is not None: neg.append(v)
        pos=[score(r.get("detection_wm",{}),ch) for r in rows]
        pos=[v for v in pos if v is not None]
        m=metrics(neg,pos)
        if m: out[(method,condition,ch,"Before Attack")]=m
        attacks=sorted({k for r in rows for k in (r.get("attacks") or {}).keys()})
        for atk in attacks:
            ap=[]
            for r in rows:
                rec=(r.get("attacks") or {}).get(atk,{})
                v=score(rec.get("detection_results",{}),ch)
                if v is not None: ap.append(v)
            am=metrics(neg,ap)
            if am: out[(method,condition,ch,atk)]=am
    return out

def main():
    a=parse_args()
    white=discover_files(a.white_dir,"white")
    no_white=discover_files(a.no_white_dir,"no_white")
    methods=sorted(set(white)|set(no_white))
    if not methods: raise RuntimeError("No *_white.jsonl or *_no_white.jsonl files found.")
    allm={}
    for method,path in white.items():
        allm.update(summarize_file(path,method,"white",a.negative_source))
    for method,path in no_white.items():
        allm.update(summarize_file(path,method,"no_white",a.negative_source))
    keys=sorted({(m,ch,atk) for (m,c,ch,atk) in allm})
    output=[]
    for method,ch,atk in keys:
        w=allm.get((method,"white",ch,atk)); nw=allm.get((method,"no_white",ch,atk))
        # One row per method/channel/attack, both conditions side by side; deltas = white - no_white.
        row={"method":method,"channel":ch,"attack":atk}
        for cond,label in ((nw,"no_whitening"),(w,"whitening")):
            for metric in ("AUROC","TPR@1%FPR","TPR@5%FPR","n_neg","n_pos"):
                row[f"{label}_{metric}"]=cond.get(metric) if cond else None
        for metric in ("AUROC","TPR@1%FPR","TPR@5%FPR"):
            row[f"delta_{metric}"]=(w[metric]-nw[metric]) if w and nw else None
        output.append(row)
    out=Path(a.output_csv).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    fields=["method","channel","attack",
        "no_whitening_AUROC","whitening_AUROC","delta_AUROC",
        "no_whitening_TPR@1%FPR","whitening_TPR@1%FPR","delta_TPR@1%FPR",
        "no_whitening_TPR@5%FPR","whitening_TPR@5%FPR","delta_TPR@5%FPR",
        "no_whitening_n_neg","whitening_n_neg","no_whitening_n_pos","whitening_n_pos"]
    # Flatten the metrics into intuitive paired columns.
    with open(out,"w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for r in output:
            flat={"method":r["method"],"channel":r["channel"],"attack":r["attack"]}
            for metric in ("AUROC","TPR@1%FPR","TPR@5%FPR"):
                flat[f"no_whitening_{metric}"]=r.get(f"no_whitening_{metric}")
                flat[f"whitening_{metric}"]=r.get(f"whitening_{metric}")
                flat[f"delta_{metric}"]=r.get(f"delta_{metric}")
            for metric in ("n_neg","n_pos"):
                flat[f"no_whitening_{metric}"]=r.get(f"no_whitening_{metric}")
                flat[f"whitening_{metric}"]=r.get(f"whitening_{metric}")
            writer.writerow(flat)
    print(f"[OK] Saved: {out}")
    print(f"[INFO] methods with whitening files={sorted(white)}")
    print(f"[INFO] methods with no-whitening files={sorted(no_white)}")

if __name__=="__main__": main()
