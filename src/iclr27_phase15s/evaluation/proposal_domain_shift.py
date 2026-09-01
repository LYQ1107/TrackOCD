"""One preregistered input/proposal domain-shift diagnostic for S-D."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def stats(rows, feats, ann, role):
    imgs = {int(x["id"]): x for x in ann["images"]}; vals=[]; scores=[]; ious=[]; labels=[]
    for i, r in enumerate(rows):
        if role is not None and r.get("gt_role") != role: continue
        im=imgs.get(int(r.get("image_id",-1))); 
        if im is None: continue
        b=json.loads(r["bbox_xyxy"]); area=max(0.,(b[2]-b[0])*(b[3]-b[1])); vals.append(area/max(float(im.get("width",1))*float(im.get("height",1)),1.)); scores.append(float(r.get("score",0.)))
        if r.get("gt_temporal_iou") not in (None,""): ious.append(float(r["gt_temporal_iou"]))
        elif r.get("gt_iou") not in (None,""): ious.append(float(r["gt_iou"]))
        labels.append(int(r.get("gt_category_id",-1)))
    a=np.asarray(vals,float); s=np.asarray(scores,float); q=np.asarray(ious,float)
    return {"rows":len(vals),"area_fraction":{"mean":float(a.mean()) if len(a) else None,"median":float(np.median(a)) if len(a) else None,"q10":float(np.quantile(a,.1)) if len(a) else None,"q90":float(np.quantile(a,.9)) if len(a) else None},"proposal_score":{"mean":float(s.mean()) if len(s) else None,"median":float(np.median(s)) if len(s) else None},"alignment_iou":{"mean":float(q.mean()) if len(q) else None,"median":float(np.median(q)) if len(q) else None},"categories":len(set(labels)),"category_counts":{str(k):int(v) for k,v in Counter(labels).items()}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--public-proposals",required=True);ap.add_argument("--public-annotation",required=True);ap.add_argument("--devplus-proposals",required=True);ap.add_argument("--devplus-annotation",required=True);ap.add_argument("--public-features",required=True);ap.add_argument("--devplus-features",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
    pr=list(csv.DictReader((ROOT/args.public_proposals).open())); dr=list(csv.DictReader((ROOT/args.devplus_proposals).open())); pa=json.load((ROOT/args.public_annotation).open()); da=json.load((ROOT/args.devplus_annotation).open()); pf=np.load(ROOT/args.public_features,allow_pickle=False); df=np.load(ROOT/args.devplus_features,allow_pickle=False)
    value={"protocol":"trackocd_iclr27_phase15s16_S-D","diagnostic_only":True,"model_selection_or_threshold_fit":False,"devplus_labels_used_for_fit":False,"q1_label_used":False,"public_known_bank_role":stats([r for r in pr if r.get('gt_role')=='known'],pf['roi'],pa,'known'),"devplus_supported_known":stats(dr,df['roi'],da,'supported_known'),"public_all_proposals":stats(pr,pf['roi'],pa,None),"devplus_all_proposals":stats(dr,df['roi'],da,None),"representation_mean_cosine_public_devplus":float(np.mean(np.asarray(pf['roi']).mean(axis=0) @ np.asarray(df['roi']).mean(axis=0))),"source_families":{"public":dict(Counter(r.get('source_family','') for r in pr)),"devplus":dict(Counter(r.get('source_family','') for r in dr))},"interpretation":"bounded proposal/input diagnostic only; no automatic backbone swap"}
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True));tmp.replace(out);print(json.dumps(value,indent=2,sort_keys=True))


if __name__=='__main__': main()
