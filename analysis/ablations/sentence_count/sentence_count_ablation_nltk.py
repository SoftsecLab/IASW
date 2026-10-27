#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence-count ablation on existing records. No generation; source JSONL is read-only."""
import argparse, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import nltk
from sklearn.metrics import roc_auc_score, roc_curve

# Use the same NLTK sentence tokenizer as the IAS-W detector configuration.
NLTK_DATA_PATH = os.environ.get("NLTK_DATA", "/home/share/models/nltk_data")
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

def sents(text):
    text = (text or "").strip()
    if not text:
        return []
    # Deliberately use NLTK's default English sentence tokenizer, not a regex splitter.
    return [s.strip() for s in nltk.sent_tokenize(text, language="english") if s.strip()]

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def parse_scores(r):
    if not isinstance(r, dict) or not r or "error" in r: return {}
    dual = any(k in r for k in ("z_score_micro","z_score_macro","z_score_fisher","micro","macro"))
    if not dual and "z_score" in r: return {"score": float(r["z_score"])}
    m = r.get("z_score_micro", 0.0)
    if m == 0 and isinstance(r.get("micro"), dict): m = r["micro"].get("z_score", 0.0)
    a = r.get("z_score_macro", 0.0)
    if a == 0 and isinstance(r.get("macro"), dict): a = r["macro"].get("z_score", 0.0)
    vals = {"kgw":m, "iasw":r.get("z_score_macro_calibrated",a),
            "fisher":r.get("z_score_fisher"), "max":r.get("z_score_max")}
    out={}
    for k,v in vals.items():
        try:
            if v is not None and math.isfinite(float(v)): out[k]=float(v)
        except (ValueError,TypeError): pass
    return out

def calc(neg,pos):
    neg=np.asarray(neg,float); pos=np.asarray(pos,float)
    if not len(neg) or not len(pos):
        return {"n_negative":len(neg),"n_positive":len(pos),"auroc":np.nan,"tpr_at_1pct_fpr":np.nan,"tpr_at_5pct_fpr":np.nan}
    y=np.r_[np.zeros(len(neg),int),np.ones(len(pos),int)]; z=np.r_[neg,pos]
    fpr,tpr,_=roc_curve(y,z)
    def at(q):
        ix=np.flatnonzero(fpr<=q); return float(tpr[ix[-1]]) if len(ix) else 0.
    return {"n_negative":len(neg),"n_positive":len(pos),"auroc":float(roc_auc_score(y,z)),
            "tpr_at_1pct_fpr":at(.01),"tpr_at_5pct_fpr":at(.05)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input_file",required=True); p.add_argument("--output_dir",required=True)
    p.add_argument("--model_path",required=True); p.add_argument("--embedder_path",default="/home/share/models/bge-small-en-v1.5")
    p.add_argument("--dataset_type",default="c4"); p.add_argument("--method",default="universal_dual",choices=["universal_dual","iasw","ias"])
    p.add_argument("--micro_method",default="kgw"); p.add_argument("--macro_method",default="iasw")
    p.add_argument("--secret_key",default="DUAL_MACRO_2026"); p.add_argument("--device",default="cuda:0")
    p.add_argument("--sentence_counts",default="1,2,3,4,5,6"); p.add_argument("--num_samples",type=int,default=0)
    p.add_argument("--target_len",type=int,default=200); p.add_argument("--negative_source",choices=["human","no_wm"],default="human")
    p.add_argument("--cohort",choices=["nested","fixed"],default="fixed")
    p.add_argument("--stats_key",default="rank64_eps0p0001_calibskip3000_calibn1000")
    p.add_argument("--alignment_mode",default="dp_local")
    p.add_argument("--boundary_tpr1",type=float,default=.80,help="Operational TPR@1%%FPR target; adjust to the application")
    p.add_argument("--boundary_tpr5",type=float,default=.90,help="Supporting TPR@5%%FPR target; adjust to the application")
    a=p.parse_args(); ks=sorted(set(int(x) for x in a.sentence_counts.split(",") if x.strip()))
    if not ks or min(ks)<1: p.error("sentence_counts must be positive integers")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    os.environ.update(IAS_STATS_KEY=a.stats_key,IASW_STATS_KEY=a.stats_key,
                      IAS_ALIGNMENT_MODE=a.alignment_mode,IASW_ALIGNMENT_MODE=a.alignment_mode,
                      IAS_TARGET_LEN=str(a.target_len))
    sys.path.insert(0,str(Path.cwd()))
    from src.watermarks.plugins import get_watermark_class
    from src.watermarks.runtime.loaders import load_runtime
    rows=read_jsonl(a.input_file)
    if a.num_samples>0: rows=rows[:a.num_samples]
    neg_field = "human_text" if a.negative_source == "human" else "no_wm_text"
    fields=("wm_text", neg_field); usable=[]
    for i,r in enumerate(rows):
        if all(isinstance(r.get(f),str) and r[f].strip() for f in fields):
            r["_n"]={f:len(sents(r[f])) for f in fields}; r["_id"]=r.get("sample_id",r.get("idx",r.get("id",i))); usable.append(r)
    if not usable: raise RuntimeError(f"Input must contain non-empty wm_text and {neg_field} fields.")
    if a.cohort=="fixed":
        mx=max(ks); usable=[r for r in usable if min(r["_n"].values())>=mx]
        if not usable: raise RuntimeError(f"No records have at least {mx} sentences in wm_text and {neg_field}; lower max K or use nested cohort.")
    args=SimpleNamespace(task="detect",method=a.method,model_path=a.model_path,embedder_path=a.embedder_path,
        baseline_file="",target_file=a.input_file,dataset_type=a.dataset_type,num_samples=a.num_samples,mode="baseline_eval",
        attack_name="",max_total_tokens=a.target_len,max_sent_tokens=40,num_candidates=32,top_k=50,
        candidate_selection_policy="macro",candidate_selection_seed=20260823,generation_seed=-1,
        dual_temperature=1.,dual_top_k=-1,dual_top_p=-1.,dual_stagea_candidates=-1,
        candidate_audit_file="",candidate_audit_max_samples=0,candidate_audit_save_text=False,
        secret_key=a.secret_key,device=a.device,micro_method=a.micro_method,macro_method=a.macro_method,
        timeout=300,gamma=.5,delta=2.,entropy_threshold=.9,dualmark_reference_opt_path="/home/share/models/opt-1.3b",
        synguard_config="",synguard_embedding_path="/home/share/models/compositional-bert-large-uncased")
    needs_model=(a.method=="universal_dual" and a.micro_method in {"sweet","dualmark"}) or a.method in {"sweet","dualmark"}
    needs_emb=a.method in {"iasw","ias","universal_dual"} and (a.method!="universal_dual" or a.macro_method in {"iasw","ias","ias_w"})
    rt=load_runtime(args,task="detect",load_model=needs_model,load_embedder=needs_emb)
    engine=get_watermark_class(a.method)(args,rt.tokenizer,rt.model,rt.embedder)
    details=[]; summary=[]
    for k in ks:
        pool=usable if a.cohort=="fixed" else [r for r in usable if min(r["_n"].values())>=k]
        channels={}; nscored=0
        for r in pool:
            tx={}
            for f in fields:
                ss=sents(r[f])
                if len(ss)<k: break
                tx[f]=" ".join(ss[:k])
            if len(tx)!=2: continue
            negtext=tx[neg_field]
            try: ns=parse_scores(engine.detect(negtext)); ps=parse_scores(engine.detect(tx["wm_text"]))
            except Exception as e:
                print(f"[WARN] K={k} id={r['_id']}: {e}",flush=True); continue
            for ch in set(ns)&set(ps):
                channels.setdefault(ch,{"neg":[],"pos":[]})
                channels[ch]["neg"].append(ns[ch]); channels[ch]["pos"].append(ps[ch])
                details.append({"sentence_count":k,"sample_id":r["_id"],"channel":ch,
                    "negative_source":a.negative_source,"negative_score":ns[ch],"watermarked_score":ps[ch],
                    "original_wm_sentence_count":r["_n"]["wm_text"]})
            nscored+=1
        for ch,d in channels.items():
            summary.append({"dataset":a.dataset_type,"model":Path(a.model_path).name,"sentence_count":k,"channel":ch,"cohort":a.cohort,"records_scored":len(d["pos"]),
                            "negative_source":a.negative_source,**calc(d["neg"],d["pos"])})
        print(f"[K={k}] records_scored={nscored}",flush=True)
    df=pd.DataFrame(summary); df.to_csv(out/"sentence_count_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(out/"sentence_count_sample_scores.csv",index=False,encoding="utf-8-sig")
    # Empirical minimum K under explicit, adjustable operating criteria.
    boundary_rows=[]
    if not df.empty:
        for ch, sub in df.groupby("channel"):
            sub=sub.sort_values("sentence_count")
            passed=sub[(sub["tpr_at_1pct_fpr"] >= a.boundary_tpr1) & (sub["tpr_at_5pct_fpr"] >= a.boundary_tpr5)]
            boundary_rows.append({"dataset":a.dataset_type,"model":Path(a.model_path).name,"channel":ch,
                "criterion_tpr1":a.boundary_tpr1,"criterion_tpr5":a.boundary_tpr5,
                "min_sentence_count_meeting_both":int(passed.iloc[0]["sentence_count"]) if len(passed) else None,
                "criteria_met":bool(len(passed)),"interpretation":"point-estimate boundary; not a universal threshold"})
    pd.DataFrame(boundary_rows).to_csv(out/"sentence_count_boundary.csv",index=False,encoding="utf-8-sig")
    (out/"run_config.json").write_text(json.dumps(vars(a),ensure_ascii=False,indent=2),encoding="utf-8")
    if not df.empty:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric,label,name in [("tpr_at_1pct_fpr","TPR@1%FPR (%)","tpr1_by_sentence_count.png"),
                                  ("tpr_at_5pct_fpr","TPR@5%FPR (%)","tpr5_by_sentence_count.png"),
                                  ("auroc","AUROC (%)","auroc_by_sentence_count.png")]:
            fig,ax=plt.subplots(figsize=(8,5))
            for ch,sub in df.groupby("channel"):
                sub=sub.sort_values("sentence_count"); ax.plot(sub.sentence_count,sub[metric]*100,marker="o",label=ch)
            ax.set_xlabel("Number of sentences"); ax.set_ylabel(label); ax.set_xticks(sorted(df.sentence_count.unique()))
            ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/name,dpi=200); plt.close(fig)
    print(f"[DONE] {out.resolve()}")
    print("  sentence_count_metrics.csv; sentence_count_sample_scores.csv; sentence_count_boundary.csv")

if __name__=="__main__": main()
