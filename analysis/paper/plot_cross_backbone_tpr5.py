#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from .build_paper import MAIN_ROOT, PAPER_FIGURES, set_data_root
except ImportError:
    from build_paper import MAIN_ROOT, PAPER_FIGURES, set_data_root
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

DATASET="c4"
ATTACKS=["typo","drop","swap","del_ins","pegasus","dipper","back_translation","copy_paste","sentence_reorder"]
ATTACK_TO_JSON_KEY={"typo":"After Typo","drop":"After Drop","swap":"After Swap","del_ins":"After Del_ins","pegasus":"After Pegasus","dipper":"After Dipper","back_translation":"After Back_translation","copy_paste":"After Copy_paste","sentence_reorder":"After Sentence_reorder"}
MODEL_ORDER=["opt-1.3b-semstamp","Llama-3.2-3B-Instruct","Mistral-7B-Instruct","qwen2.5-7b-instruct","llama3.1-8b-instruct"]
MODEL_DISPLAY={"opt-1.3b-semstamp":"OPT-SemStamp","Llama-3.2-3B-Instruct":"Llama-3.2-3B","Mistral-7B-Instruct":"Mistral-7B","qwen2.5-7b-instruct":"Qwen2.5-7B","llama3.1-8b-instruct":"Llama-3.1-8B"}
METHOD_ORDER=["kgw","sta","sweet","dipmark","iasw","semstamp","ksemstamp","simmark_cosine","simmark_euclidean","dualmark","synguard","dual_kgw_iasw"]
METHOD_DISPLAY={"kgw":"KGW","sta":"STA","sweet":"SWEET","dipmark":"DiPmark","iasw":"IAS-W","semstamp":"SemStamp","ksemstamp":"k-SemStamp","simmark_cosine":"Cos.-SimMark","simmark_euclidean":"Euc.-SimMark","dualmark":"DualMark","synguard":"SynGuard","dual_kgw_iasw":"KGW+IAS-W"}
SEMANTIC={"iasw","semstamp","ksemstamp","simmark_cosine","simmark_euclidean"}
OURS="dual_kgw_iasw"
KGW="kgw"
_COLOR_POOL=plt.get_cmap("tab20").colors
METHOD_COLORS={m:_COLOR_POOL[i] for i,m in enumerate(METHOD_ORDER)}
METRICS_SUFFIX="_roc_combined_metrics.json"
FIGSIZE=(6.3,4.6)
LEGEND_NCOL=4
LEGEND_Y=-0.255
BOTTOM=0.34

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default=None,
                   help="Released data/ directory (flat records layout).")
    p.add_argument("--font-size",type=float,default=11.5)
    p.add_argument("--label-size",type=float,default=12.4)
    p.add_argument("--tick-size",type=float,default=11.0)
    p.add_argument("--legend-size",type=float,default=10.2)
    p.add_argument("--line-width",type=float,default=1.15)
    p.add_argument("--kgw-line-width",type=float,default=1.55)
    p.add_argument("--ours-line-width",type=float,default=2.45)
    p.add_argument("--marker-size",type=float,default=4.2)
    p.add_argument("--ours-marker-size",type=float,default=7.8)
    p.add_argument("--tiff-dpi",type=int,default=600)
    p.add_argument("--show-title",action="store_true")
    p.add_argument("--no-eps",action="store_true")
    p.add_argument("--no-tiff",action="store_true")
    return p.parse_args()
def configure(args):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": args.font_size,
        "axes.labelsize": args.label_size,
        "xtick.labelsize": args.tick_size,
        "ytick.labelsize": args.tick_size,
        "legend.fontsize": args.legend_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
    })
def save(fig,args):
    for ext in ["pdf"] + ([] if args.no_eps else ["eps"]) + ([] if args.no_tiff else ["tiff"]):
        p=PAPER_FIGURES/f"figure4_cross_backbone_tpr5_line.{ext}"
        kw={"bbox_inches":"tight"}
        if ext=="tiff": kw.update(format="tiff",dpi=args.tiff_dpi,pil_kwargs={"compression":"tiff_lzw"})
        elif ext=="eps": kw.update(format="eps")
        fig.savefig(p,**kw); print(f"[DONE] {p}")
def base(path):
    return path.name[:-len(METRICS_SUFFIX)] if path.name.endswith(METRICS_SUFFIX) else path.stem
def infer(s):
    s=s.lower()
    if s.startswith("dual_kgw_iasw"): return "dual_kgw_iasw"
    if s=="iasw" or s.startswith("iasw_") or s=="iaswa" or s.startswith("iaswa_"): return "iasw"
    for k in ["simmark_euclidean","simmark_cosine","ksemstamp","semstamp","dualmark","synguard","dipmark","sweet","sta","kgw"]:
        if s==k or s.startswith(k+"_"): return k
    return None
def preference(path,key):
    b=base(path).lower(); score=1000 if b==key else 0
    if key in {"iasw","dual_kgw_iasw"}:
        for tag in ["rank64","cand32","dp_local","eps0p0001","calibskip3000","calibn1000"]:
            if tag in b: score+=20
    return score,-len(b)
def discover(folder):
    cand={}
    if not folder.exists(): return {}
    for p in folder.glob(f"*{METRICS_SUFFIX}"):
        k=infer(base(p))
        if k in METHOD_ORDER: cand.setdefault(k,[]).append(p)
    return {k:max(v,key=lambda p:preference(p,k)) for k,v in cand.items()}
def choose(report,key):
    pref=["Fisher Fusion","Fisher","Micro","Macro"] if key==OURS else (["Macro","Micro","Fisher Fusion"] if key in SEMANTIC else ["Micro","Fisher Fusion","Macro"])
    for n in pref:
        sec=report.get(n)
        if isinstance(sec,dict) and any(isinstance(v,dict) and "TPR@5%FPR" in v for v in sec.values()): return sec
    return None
def avg(path,key):
    report=json.load(open(path,encoding="utf-8")); sec=choose(report,key)
    if sec is None: return np.nan
    vals=[]
    for a in ATTACKS:
        item=sec.get(ATTACK_TO_JSON_KEY[a])
        if not isinstance(item,dict): return np.nan
        try: v=float(item["TPR@5%FPR"])
        except: return np.nan
        vals.append(v)
    return float(np.mean(vals))
def main():
    global MAIN_ROOT
    args=parse_args()
    if args.data_root:
        import sys
        set_data_root(args.data_root)
        _bp = (sys.modules.get("build_paper")
               or sys.modules.get(f"{__package__}.build_paper"))
        MAIN_ROOT = _bp.MAIN_ROOT
    configure(args)
    per_model={m:discover(MAIN_ROOT/DATASET/m/"plots") for m in MODEL_ORDER}
    x=np.arange(len(MODEL_ORDER)); labels=[MODEL_DISPLAY[m] for m in MODEL_ORDER]
    fig,ax=plt.subplots(figsize=FIGSIZE)
    for key in METHOD_ORDER:
        y=[]
        for m in MODEL_ORDER:
            p=per_model[m].get(key); y.append(avg(p,key)*100 if p else np.nan)
        if not np.isfinite(y).any(): continue
        ours=key==OURS
        is_kgw=key==KGW
        linewidth=args.ours_line_width if ours else (args.kgw_line_width if is_kgw else args.line_width)
        alpha=1.0 if (ours or is_kgw) else 0.76
        ax.plot(
            x,y,
            marker="*" if ours else "o",
            color=METHOD_COLORS[key],
            linewidth=linewidth,
            markersize=args.ours_marker_size if ours else args.marker_size,
            alpha=alpha,
            label=METHOD_DISPLAY[key],
            zorder=12 if ours else (7 if is_kgw else 3),
        )
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, ha="center"); ax.set_xlabel("Backbone"); ax.set_ylabel("Average TPR@5%FPR (%)")
    if args.show_title: ax.set_title("Cross-model Robustness — Average TPR@5%FPR (RealNews)")
    ax.set_ylim(0,100); ax.set_yticks(np.arange(0,101,20))
    ax.grid(axis="y",linestyle="--",linewidth=.65,alpha=.22)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,LEGEND_Y),ncol=LEGEND_NCOL,frameon=False,handlelength=1.8,handletextpad=.45,columnspacing=.85,borderaxespad=0.0)
    fig.subplots_adjust(left=.13,right=.98,top=.95 if not args.show_title else .89,bottom=BOTTOM)
    save(fig,args); plt.close(fig)
if __name__=="__main__": main()
