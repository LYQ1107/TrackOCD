#!/usr/bin/env python3
"""Plot actual sealed Train loss traces; no training or Val metrics access."""
import csv
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def trace_rows(family,fit):
    rows=[]
    for row in fit['trace']:
        if family=='policy':
            loss=sum(e['loss'] for e in row['episodes'])/len(row['episodes']);step=row['epoch']
            phase='T3';context=fit['representation']
        else:loss=row['total_loss'];step=row['step'];phase='T1' if family=='representation' else 'T2';context='adapter_and_extra_module'
        rows.append({'phase':phase,'context':context,'model':fit['model'],'seed':fit['seed'],'step_or_epoch':step,'actual_training_loss':loss})
    return rows


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file
    base=ROOT/'outputs/trackocd_core/core_training/train_first_v1';out=ROOT/'outputs/trackocd_core';rows=[];fits=[];lineages={}
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    colors={1027:'#2460a7',1028:'#db7727',1029:'#299869'}
    for family in ('representation','evidence','policy'):
        p=base/family/'training_receipt.json';d=json.loads(p.read_text());lineages[family]={'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p),'preregistration_commit':d['preregistration_commit']}
        for fit in d['fits']:
            trace=trace_rows(family,fit);rows.extend(trace)
            selected=fit['selected_checkpoint'].get('step',fit['selected_checkpoint'].get('epoch'))
            fits.append({'family':family,'model':fit['model'],'representation':fit.get('representation'),'seed':fit['seed'],
                'recorded_steps_or_epochs':len(trace),'first_loss':trace[0]['actual_training_loss'],'last_loss':trace[-1]['actual_training_loss'],
                'selected_step_or_epoch':selected,'checkpoint':fit['selected_checkpoint']['checkpoint']})
            if family=='representation':
                column={'A1_ADAPTER':0,'A1_GEOMETRY_1':1,'A1_GEOMETRY_5':2}[fit['model']];ax=axes[0,column];label=str(fit['seed']);ax.set_title(fit['model'])
            elif family=='evidence':continue
            else:
                column={'A0_RAW':0,'A1_SELECTED':1,'A2_EVIDENCE':2}[fit['representation']];ax=axes[1,column]
                label=f"{fit['model'].split('_')[0]} / {fit['seed']}";ax.set_title(fit['representation']+' policy train objective')
            values=np.asarray([r['actual_training_loss'] for r in trace]);x=np.asarray([r['step_or_epoch'] for r in trace])
            if family=='representation':
                # Fixed25step display smoothing; unchanged actual traces inCSV.
                values=np.convolve(values,np.ones(25)/25,mode='valid');x=x[24:]
            ax.plot(x,values,color=colors[fit['seed']],linestyle='--' if fit['model'].startswith('D2') else '-',label=label,linewidth=1)
    for i,ax in enumerate(axes.flat):
        ax.set_xlabel('step (fixed25step display mean)' if i<3 else 'epoch');ax.set_ylabel('actual Train loss');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.suptitle('Real Train traces; D1/D2 objectives differ, losses are not comparative test scores',fontsize=10)
    figure=out/'TRAINING_LOSS_CURVES.png';fig.savefig(figure,dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for fit in json.loads((base/'evidence/training_receipt.json').read_text())['fits']:
        trace=trace_rows('evidence',fit);v=np.asarray([r['actual_training_loss'] for r in trace]);v=np.convolve(v,np.ones(25)/25,mode='valid')
        ax.plot([r['step_or_epoch'] for r in trace][24:],v,color=colors[fit['seed']],linestyle='--' if fit['model']=='A2_CAPACITY_CONTROL' else '-',label=f"{fit['model']} / {fit['seed']}",linewidth=1)
    ax.set(xlabel='step (fixed25step display mean)',ylabel='actual Train loss',title='Temporal vs same-capacity static training, not heldout superiority')
    ax.legend(fontsize=7);ax.grid(alpha=.2);evidence_figure=out/'EVIDENCE_TRAINING_LOSS_CURVES.png';fig.savefig(evidence_figure,dpi=140);plt.close(fig)
    buffer=io.StringIO();w=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    path=out/'TRAINING_LOSS_TRACES.csv';atomic_write_text(path,buffer.getvalue())
    assert len(fits)==33 and len(rows)==15360
    result={'status':'COMPLETE_ACTUAL_TRAIN_LOSS_TRACE_REPORT','actual_fits':33,'actual_trace_rows':len(rows),'receipt_lineages':lineages,'fits':fits,
        'artifacts':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in (figure,evidence_figure,path)},
        'source_sha256':sha256_file(Path(__file__).resolve()),'loss_is_not_heldout_metric_or_scientific_PASS':True,'D1_D2_have_different_objectives':True,
        'display_smoothing_only':25,'new_training':False,'Val_or_Test_read':False}
    atomic_json(out/'TRAINING_CURVES_RESULT.json',result);print(json.dumps({'status':result['status'],'actual_fits':33,'trace_rows':len(rows)}))


if __name__=='__main__':main()
