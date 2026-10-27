#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build separate no-whitening IAS-W stats; does not edit project source or source stats."""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument("--source_stats", required=True)
    p.add_argument("--baseline_file", required=True)
    p.add_argument("--output_stats", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--embedder_path", required=True)
    p.add_argument("--dataset_type", default="c4")
    p.add_argument("--target_len", type=int, default=200)
    p.add_argument("--secret_key", default="DUAL_MACRO_2026")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_calibration_texts", type=int, default=1000,
                   help="Maximum human_text/no_wm_text strings combined; <=0 uses all.")
    p.add_argument("--min_calibration_texts", type=int, default=100)
    return p.parse_args()

def read_jsonl(path):
    out=[]
    with open(path, encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except Exception as e: raise RuntimeError(f"Invalid JSON {path}:{i}: {e}") from e
    return out

def main():
    a=args_parse()
    root=Path.cwd().resolve()
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    src=Path(a.source_stats).expanduser().resolve()
    base=Path(a.baseline_file).expanduser().resolve()
    out=Path(a.output_stats).expanduser().resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    if not base.is_file(): raise FileNotFoundError(base)
    try: stats0=torch.load(src,map_location="cpu",weights_only=False)
    except TypeError: stats0=torch.load(src,map_location="cpu")
    needed=("mean","components","eigvals","calibration","calibration_profiles")
    missing=[k for k in needed if k not in stats0]
    if missing: raise KeyError(f"Source stats missing {missing}: {src}")
    dim=int(stats0["components"].shape[0])
    stats=copy.deepcopy(stats0)
    # Identity projection + zero mean. Division by sqrt(1+eps) disappears after L2 normalization.
    stats["mean"]=torch.zeros(dim,dtype=torch.float32)
    stats["components"]=torch.eye(dim,dtype=torch.float32)
    stats["eigvals"]=torch.ones(dim,dtype=torch.float32)
    stats["rank"]=dim
    stats["whiten_dim"]=dim
    stats["original_dim"]=dim
    stats["no_whitening_ablation"]={"source_stats":str(src),"baseline_file":str(base),
       "transform":"normalized raw embedding; zero mean; identity projection"}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".uncalibrated.tmp.pt")
    torch.save(stats,tmp)
    rows=read_jsonl(base)
    texts=[]
    for row in rows:
        for key in ("human_text","no_wm_text"):
            x=row.get(key)
            if isinstance(x,str) and x.strip():
                texts.append(x.strip())
                if a.max_calibration_texts>0 and len(texts)>=a.max_calibration_texts: break
        if a.max_calibration_texts>0 and len(texts)>=a.max_calibration_texts: break
    if len(texts)<a.min_calibration_texts:
        raise RuntimeError(f"Only {len(texts)} calibration texts; need >= {a.min_calibration_texts}.")
    from src.watermarks.ias_w_mark import IASWMark
    embedder=SentenceTransformer(a.embedder_path).to(a.device)
    profiles=stats.get("calibration_profiles",{})
    if not isinstance(profiles,dict): raise TypeError("calibration_profiles must be a dict")
    for profile in ("standalone","dual"):
        pack=profiles.get(profile)
        if not isinstance(pack,dict):
            raise RuntimeError(f"Source stats lack calibration_profiles['{profile}']; cannot run both methods safely.")
        cal=pack.setdefault("calibration",{})
        if not isinstance(cal,dict): raise TypeError(f"{profile} calibration must be a dict")
        mark=IASWMark(embedder_path_or_model=embedder,tokenizer=None,
            model_name_path=a.model_path,dataset_type=a.dataset_type,secret_key=a.secret_key,
            device=a.device,stats_file=str(tmp),target_len=a.target_len,scoring_profile=profile)
        vals=[]; errors=0
        for text in tqdm(texts,desc=f"Calibrating {profile}"):
            try:
                score=float(mark._compute_raw_scores(text).get("raw_scores",{}).get("dp_local",float("nan")))
                if np.isfinite(score): vals.append(score)
                else: errors+=1
            except Exception as e:
                errors+=1
                if errors<=3: print(f"[WARN] {profile} calibration: {type(e).__name__}: {e}")
        if len(vals)<a.min_calibration_texts:
            raise RuntimeError(f"{profile}: only {len(vals)} valid scores; no final stats written.")
        arr=np.asarray(vals,dtype=np.float64); sigma=float(arr.std(ddof=0))
        if not np.isfinite(sigma) or sigma<=1e-8: raise RuntimeError(f"{profile}: invalid sigma {sigma}")
        cal["dp_local"]={"mu":float(arr.mean()),"sigma":sigma,"n":int(arr.size)}
        print(f"[OK] {profile}: mu={arr.mean():.6f}, sigma={sigma:.6f}, n={len(arr)}")
        del mark
    stats["calibration_profiles"]=profiles
    torch.save(stats,out)
    report={"source_stats":str(src),"output_stats":str(out),"baseline_file":str(base),
        "dataset_type":a.dataset_type,"target_len":a.target_len,"original_dim":dim,
        "whiten_dim":dim,"num_calibration_texts":len(texts),
        "calibration_profiles":{p:profiles[p]["calibration"].get("dp_local") for p in ("standalone","dual")}}
    out.with_suffix(out.suffix+".json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.unlink(missing_ok=True)
    print(f"[OK] Saved no-whitening stats: {out}")

if __name__=="__main__": main()
