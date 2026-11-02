"""Rebuild the main text's displayed numeric tables."""
import csv,json,re

def build(ev,out):
    plans=json.loads((ev.base/'records/main_layouts.json').read_text())
    pair=lambda v:f'{v["strict"]:,} / {v["semantic"]:,}'
    for name,p in plans.items():
        rs=[]
        for row in p['rows']:
            r=dict(row);m=r['Setting'];d=int(re.match(r'\d+',r.get('Evaluation depth','') or str(ev.primary[m]))[0])
            if name=='main_02':
                t=ev.config[m]['training'];T=t['total_updates'];b=t.get('effective_batch',t['physical_batch']);a=t['act_steps'];N=ev.config[m]['data']['train']['count']
                r.update(**{'Training rows':f'{N:,}','Batch':f'{b:,}','Depth':str(ev.primary[m]),'Updates':f'{T:,}','Fresh presentations':f'{T*b//a:,}','Participations':f'{T*b:,}','GPU-h':f'{ev.charges[m]/3600:.2f}'})
            elif name=='main_03':
                u,a=ev.endpoint(m,d);v,b=ev.endpoint(m,d,'best')
                r.update(**{'Final strict / semantic':f'{a["strict_correct"]:,} / {a["semantic_correct"]:,}','Best strict / semantic':f'{b["strict_correct"]:,} / {b["semantic_correct"]:,}','Best update':f'{v:,}','Evaluations':str(len(ev.evals[m]))})
            elif name=='main_04':
                for col,sp in [('Original training sample','original_train'),('Added training sample','new_train'),('CPU development','id_dev')]:
                    if r[col]!='Not scored':r[col]=pair(ev.score(m+'_final',d,sp))
            elif name=='main_05':
                sp='original_train' if r['Source sample']=='Original' else 'new_train';a=ev.score(m+'_final',8,sp+'_seen');b=ev.score(m+'_final',8,sp+'_unseen');r.update(**{'Seen strict / semantic':pair(a)+f' of {a["total"]:,}','Unseen strict / semantic':pair(b)+f' of {b["total"]:,}','Strict gap, pp':f'{100*(a["strict"]/a["total"]-b["strict"]/b["total"]):.2f}'})
            elif name=='main_06':
                for col,sp in [('CPU development','id_dev'),('Fresh ID','fresh_id'),('Fresh OOD','fresh_ood')]:r[col]=pair(ev.score(m+'_final',d,sp))
                a=ev.score(m+'_final',d,'id_dev');b=ev.score(m+'_final',d,'fresh_id');r['Gap, pp']=f'{100*(b["strict"]/b["total"]-a["strict"]/a["total"]):.2f}'
            elif name=='main_07':
                for n,col in zip(range(5,9),p['fields'][2:]):r[col]=pair(ev.score(m+'_final',d,'untouched_id','operand_count',n))
            else:raise ValueError(name)
            rs.append(r)
        with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=p['fields']);w.writeheader();w.writerows(rs)
