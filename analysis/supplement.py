"""Render displayed supplement tables from measurement records and method facts."""
import csv,json

def build(ev,out):
    def read(name):return json.loads((ev.base/'records'/name).read_text())
    def export(name,plan,rows):
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=plan['fields']);w.writeheader();w.writerows(rows)
    retention={(r['model'],r['kind']):r['retained'] for r in read('checkpoint_retention.json')}
    for name,p in read('supplement_layouts.json').items():
        result=[];kind=p['kind']
        for row in p['rows']:
            r=dict(row);m=r.get('Setting','');d=int(r.get('Depth') or ev.primary.get(m,8))
            if kind=='A02_operand_mixtures':
                n=int(r['Operands'])
                for col,ds in [('R65-resampled','training_65536'),('R131','training_131072'),('R262','training_262144'),('Full expanded file','new_train')]:r[col]=f'{ev.hist[ds][n]:,}'
            elif kind=='A03_checkpoint_retention':
                u,e=ev.endpoint(m,ev.primary[m],r['Selection']);c=next(c for c in ev.checkpoints[m] if c['update']==u)
                r.update(Update=f'{u:,}',**{'Weights retained':'Yes' if retention[m,r['Selection']]=='True' else 'No','Recorded SHA256':c['sha256']})
            elif kind=='A04_charge_exposure':
                t=ev.config[m]['training'];T=t['total_updates'];b=t.get('effective_batch',t['physical_batch']);A=t['act_steps'];N=ev.config[m]['data']['train']['count'];i={'R524':5,'R524-B4096-D16-short':6,'R262':7}.get(m)
                extra=read(f'allocations/{i:02d}_terminal.json')['charged_seconds'] if i else 0
                r.update(**{'Charged seconds':f'{ev.charges[m]:.6f}','Additional refused-root seconds':f'{extra:.6f}','Complete-pass equivalents':f'{T*b/A/N:.2f}'})
            elif kind=='A05_semantic_maxima':
                u,e=ev.endpoint(m,d,'semantic');r.update(**{'Best semantic count':f'{e["semantic_correct"]:,}','Update':f'{u:,}'})
            elif kind in ['A06_operand_fitting_gaps','A07_selection_fitting_gaps']:
                sel='best' if r.get('Selection')=='Development-selected best' else 'final';sp='original_train' if kind=='A07_selection_fitting_gaps' or r['Sample']=='Original' else 'new_train';g='operand_count' if kind=='A06_operand_fitting_gaps' else 'overall';n=r.get('Operands')
                a=ev.score(m+'_'+sel,8,sp+'_seen',g,n);b=ev.score(m+'_'+sel,8,sp+'_unseen',g,n)
                gap=100*(a['strict']/a['total']-b['strict']/b['total']) if a['total'] and b['total'] else None
                r.update(Seen=f'{a["strict"]}/{a["total"]}',Unseen=f'{b["strict"]}/{b["total"]}',**{'Gap, pp':f'{gap:.2f}' if gap is not None else 'Undefined'})
                if kind=='A07_selection_fitting_gaps':u,e=ev.endpoint(m,8,sel);r['Update']=f'{u:,}'
            elif kind=='A08_standardized_fresh':
                a=ev.standard(m+'_final',d)
                for col,key in [('Development, %','development_percent'),('Fresh, %','fresh_percent'),('Raw deficit, pp','raw_deficit_pp'),('Standardized fresh, %','standardized_fresh_percent'),('Standardized deficit, pp','standardized_deficit_pp')]:r[col]=f'{a[key]:.2f}'
            elif kind=='development_label_ambiguity':
                a=ev.ambiguity[int(r['operand_count'])];r.update(total=str(a['total']),ambiguous_rows=str(a['ambiguous_rows']),ambiguity_percent=f'{100*a["ambiguous_rows"]/a["total"]:.2f}')
            elif kind=='A09_fixtures':
                plan=read('table_layouts.json')['fixture_checks'];q=next(q for q in plan['rows'] if q['check']==r['Check']);v=ev.fill('fixture_checks',q);r['Observed maximum discrepancy']=v['maximum_discrepancy']
            elif kind=='A10_live_development':
                u,e=ev.endpoint(m,ev.primary[m]);a=ev.live[m,u];dev=100*e['strict_correct']/e['total']
                r.update(**{'Window, updates':f'{a["window_start_update"]:,}–{u:,}','Live correct / row participations':f'{a["correct"]:,} / {a["total"]:,}','Live, %':f'{a["percent"]:.2f}','Development, %':f'{dev:.2f}','Live minus development, pp':f'{a["percent"]-dev:.2f}'})
            elif kind=='A11_historical_scores':
                a=ev.score(r['Setting and checkpoint'],d,r['Population']);r.update(**{'Strict / semantic':f'{a["strict"]:,} / {a["semantic"]:,}','Rows':f'{a["total"]:,}'})
            else:raise ValueError(kind)
            result.append(r)
        export(name,p,result)
    for name,p in read('method_tables.json').items():export(name,p,p['rows'])
