"""Recompute fresh-batch control readouts from per-row predictions and run records.

Uses only Python's standard library. The sample files preserve original source
indices and row hashes. Membership comes from the retained full-file scan receipt.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
SETS=('id_dev','fresh_id','fresh_ood','untouched_id','original_train','new_train')
MODELS=('S65f','S524f')
def read(p):return json.loads(Path(p).read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(row,ops):
    if len(ops)!=len(row['operands'])-1:raise ValueError('operator count')
    stack=[];operand=operator=0
    for marker in row['structure']:
        if marker=='n':stack.append(row['operands'][operand]);operand+=1
        else:
            b=stack.pop();a=stack.pop();op=ops[operator];operator+=1
            if op==0:value=a+b
            elif op==1:value=a-b
            elif op==2:value=a*b
            elif op==3 and b and a%b==0:value=a//b
            else:raise ValueError('invalid operator/division')
            stack.append(value)
    if len(stack)!=1:raise ValueError('expression stack')
    return stack[0]
def row_hash(row):
    order=('token_ids','target_operator_ids','target_positions','target_mask','example_id','multiset','structure','operands','result','split')
    return hashlib.sha256(json.dumps({k:row[k] for k in order},separators=(',',':')).encode()).hexdigest()
def wilson(k,n):
    if not n:return None,None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-radius),min(1.,center+radius)
def aggregate(rows):
    n=len(rows);out=dict(total=n,semantic_invalid=sum(r['semantic_invalid'] for r in rows))
    for metric in ('strict','semantic'):
        k=sum(r[metric] for r in rows);lo,hi=wilson(k,n)
        out.update({metric+'_correct':k,metric+'_accuracy':k/n if n else None,metric+'_wilson_low':lo,metric+'_wilson_high':hi})
    return out

def collect(base):
    sources={sp:read(base/'records/scoring_samples'/(sp+'_rows.json')) for sp in SETS}
    membership=read(base/'records/scoring_membership.json')['models']
    groups=defaultdict(list);chunks=0
    for index in read(base/'records/prediction_index.json'):
        if index['cohort']!='collection08':continue
        path=base/index['path'];assert digest(path)==index['sha256'];chunk=read(path);item=chunk['item'];model=item['model'];sp=item['split']
        assert model.rsplit('_',1)[0] in MODELS and sp in SETS and item['depth']==1
        assert chunk['dataset_sha256']==digest(base/'records/scoring_samples'/(sp+'_rows.json'))
        predictions=chunk['predictions'];selected=sources[sp][item['start']:item['stop']]
        assert len(predictions)==len(selected)>0
        for pred,source in zip(predictions,selected):
            row=source['row'];assert (pred['source_index'],pred['example_id'],pred['row_sha256'])==(source['source_index'],row['example_id'],source['row_sha256'])
            assert source['row_sha256']==row_hash(row)
            strict=pred['pred_ops']==[x for x,active in zip(row['target_operator_ids'],row['target_mask']) if active]
            try:semantic=execute(row,pred['pred_ops'])==row['result'];invalid=False
            except (IndexError,ValueError):semantic=False;invalid=True
            assert (strict,semantic,invalid)==(pred['strict'],pred['semantic'],pred['semantic_invalid'])
            groups[model,sp].append(dict(pred,n=len(row['operands']),w=len(str(abs(row['result'])))))
        chunks+=1
    assert chunks==120
    for model in MODELS:
        for selection in ('best','final'):
            key=model+'_'+selection
            for sp in SETS:
                rows=groups[key,sp];assert len(rows)==len(sources[sp]);assert len({q['example_id'] for q in rows})==len(rows)
            fresh={q['example_id']:q for q in groups[key,'fresh_id']}
            for q in groups[key,'untouched_id']:assert q['pred_ops']==fresh[q['example_id']]['pred_ops']
            all_rows=[];all_seen=[];all_unseen=[]
            for sp in ('original_train','new_train'):
                flags=membership[model]['samples'][sp]['membership'];rows=groups[key,sp]
                assert len(flags)==len(rows)
                seen=[q for q,flag in zip(rows,flags) if flag];unseen=[q for q,flag in zip(rows,flags) if not flag]
                groups[key,sp+'_seen']=seen;groups[key,sp+'_unseen']=unseen
                all_rows+=rows;all_seen+=seen;all_unseen+=unseen
            groups[key,'combined_train']=all_rows;groups[key,'combined_train_seen']=all_seen;groups[key,'combined_train_unseen']=all_unseen
    assert len(groups)==52
    return groups

def build(base,output):
    base=Path(base);output=Path(output);output.mkdir(parents=True,exist_ok=True);groups=collect(base)
    def export(name,rows):
        assert rows
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scores=[];strata=[];gaps=[]
    for (checkpoint,split),rows in sorted(groups.items()):
        scores.append(dict(checkpoint=checkpoint,depth=1,split=split,**aggregate(rows)))
        selectors=[('operand_count',n,None) for n in range(3,9)]+[('result_width',None,w) for w in (1,2,3)]+[('joint',n,w) for n in range(3,9) for w in (1,2,3)]
        for group,n,w in selectors:
            sub=[q for q in rows if (n is None or q['n']==n) and (w is None or q['w']==w)]
            strata.append(dict(checkpoint=checkpoint,depth=1,split=split,group=group,operand_count=n,result_width=w,**aggregate(sub)))
    for model in MODELS:
        for selection in ('best','final'):
            key=model+'_'+selection;a=aggregate(groups[key,'id_dev']);b=aggregate(groups[key,'fresh_id'])
            gaps.append(dict(checkpoint=key,strict_fresh_minus_development_pp=100*(b['strict_accuracy']-a['strict_accuracy']),semantic_fresh_minus_development_pp=100*(b['semantic_accuracy']-a['semantic_accuracy'])))
    execution=[];endpoints=[];curves=[];membership=[]
    for model in MODELS:
        root=base/'records'/model;config=read(base/'configs'/(model+'.json'));t=config['training'];U=t['total_updates'];B=t['physical_batch'];A=t['act_steps']
        account=read(root/'accounting.json');terminal=read(root/'terminal.json');segments=account['segments']
        assert all(q['actual_exit']==0 and q['ended_monotonic'] is not None for q in segments)
        charge=sum(q['charged_seconds'] for q in segments)+account.get('prior_charge',{}).get('charged_seconds',0)
        assert charge==terminal['charged_seconds'];assert read(root/'manifest.json')['data_exposure']==dict(U=U,B=B,A=A,F=U*B//A)
        execution.append(dict(model=model,U=U,B=B,A=A,F=U*B//A,P=U*B,epochs=U*B//A//config['data']['train']['count'],calibration_seconds=sum(q['charged_seconds'] for q in segments if q['phase']=='calibrate'),main_seconds=sum(q['charged_seconds'] for q in segments if q['phase']=='train'),inclusive_charged_seconds=charge,inclusive_gpu_hours=charge/3600))
        evaluations=[json.loads(line) for line in (root/'curve/evaluations.jsonl').read_text().splitlines()]
        assert len(evaluations)==96 and evaluations[-1]['update']==U
        for e in evaluations:
            curves.append(dict(model=model,depth=1,update=e['update'],strict_correct=e['strict_correct'],semantic_correct=e['semantic_correct'],total=e['total']))
        for selection,e in [('final',evaluations[-1]),('best',max(evaluations,key=lambda x:x['strict_correct']))]:
            key=model+'_'+selection;cpu=aggregate(groups[key,'id_dev']);ptr=read(root/'curve'/('latest.json' if selection=='final' else 'best_ema.json'))
            assert ptr['update']==e['update']
            ptr=ptr.get('checkpoint',ptr)
            for index in read(base/'records/prediction_index.json'):
                if index['cohort']=='collection08':
                    chunk=read(base/index['path'])
                    if chunk['item']['model']==key:assert chunk['checkpoint_sha256']==ptr['sha256']
            endpoints.append(dict(checkpoint=key,depth=1,update=e['update'],recorded_strict=e['strict_correct'],recorded_semantic=e['semantic_correct'],total=e['total'],cpu_strict=cpu['strict_correct'],cpu_semantic=cpu['semantic_correct'],strict_difference=cpu['strict_correct']-e['strict_correct'],semantic_difference=cpu['semantic_correct']-e['semantic_correct'],checkpoint_sha256=ptr['sha256']))
            for sp in ('original_train','new_train','combined_train'):
                a=aggregate(groups[key,sp+'_seen']);b=aggregate(groups[key,sp+'_unseen'])
                membership.append(dict(checkpoint=key,sample=sp,seen_total=a['total'],seen_strict=a['strict_correct'],seen_semantic=a['semantic_correct'],unseen_total=b['total'],unseen_strict=b['strict_correct'],unseen_semantic=b['semantic_correct'],strict_gap_pp=100*(a['strict_accuracy']-b['strict_accuracy']) if a['total'] and b['total'] else None,semantic_gap_pp=100*(a['semantic_accuracy']-b['semantic_accuracy']) if a['total'] and b['total'] else None))
    for name,rows in [('fresh_batch_cpu_scores',scores),('fresh_batch_cpu_strata',strata),('fresh_batch_gaps',gaps),('fresh_batch_execution',execution),('fresh_batch_endpoints',endpoints),('fresh_batch_curves',curves),('fresh_batch_membership',membership)]:export(name,rows)
    print(json.dumps(dict(fresh_control_score_sets=len(groups),fresh_control_strata=len(strata),fresh_control_tables=7)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();build(args.base,args.output)
