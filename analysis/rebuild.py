"""Rebuild released tables from complete evaluation, prediction and data records."""
import argparse, collections, csv, gzip, hashlib, json, math
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def js(p):return json.loads(Path(p).read_text())
def lines(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:
  for line in f:
   if line.strip():yield json.loads(line)
def execute(row,ops):
 if len(ops)!=len(row['operands'])-1:raise ValueError('arity')
 stack=[];ni=oi=0
 for marker in row['structure']:
  if marker=='n':stack.append(row['operands'][ni]);ni+=1
  else:
   b,a=stack.pop(),stack.pop();op=ops[oi];oi+=1
   if op==0:v=a+b
   elif op==1:v=a-b
   elif op==2:v=a*b
   elif op==3 and b and a%b==0:v=a//b
   else:raise ValueError('operator or division')
   stack.append(v)
 if len(stack)!=1:raise ValueError('stack')
 return stack[0]
def wilson(k,n):
 z=1.959963984540054;a=k/n;den=1+z*z/n;c=(a+z*z/(2*n))/den;r=z*math.sqrt(a*(1-a)/n+z*z/(4*n*n))/den;return c-r,c+r
class Evidence:
 def __init__(self,base):
  self.base=base;self.config={p.stem:js(p) for p in (base/'configs').glob('*.json') if p.stem!='base' and not p.stem.endswith('_schedule')}
  self.evals={m:list(lines(base/'records'/m/'curve/evaluations.jsonl')) for m in self.config}
  self.primary={m:1 if m.startswith('S') else c['training']['act_steps'] for m,c in self.config.items()}
  self.ps=collections.defaultdict(list);self.checkpoints={m:js(base/'records'/m/'curve/checkpoints.json')['verified'] for m in self.config}
  self.best={m:(js(base/'records'/m/'curve/best_ema.json')['update'] if (base/'records'/m/'curve/best_ema.json').exists() else None) for m in self.config}
  for ix in js(base/'records/prediction_index.json'):
   p=base/ix['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==ix['sha256'],p
   if p.suffix=='.jsonl':
    model,u,split=ix['model'],ix['update'],ix['split'];preds=list(lines(p))
   else:
    chunk=js(p);item=chunk['item'];rawmodel=item['model'];model=rawmodel.removesuffix('_final').removesuffix('_best');u=self.best[model] if rawmodel.endswith('_best') else self.evals[model][-1]['update'];split=item['split'];preds=chunk['predictions']
    side=next(c for c in self.checkpoints[model] if c['update']==u)
    assert chunk['checkpoint_sha256']==side['sha256'],p
   for q in preds:
    ds={int(d):v['pred_ops'] for d,v in q['depths'].items()} if 'depths' in q else {self.primary[model]:q['pred_ops']}
    for d,ops in ds.items():self.ps[model,u,d,split,ix['cohort']].append(dict(q,ops=ops))
  wanted={p['example_id'] for qs in self.ps.values() for p in qs};self.data={};self.members=collections.defaultdict(set);self.hist={};self.indices={};self.rowhash={}
  for v in js(base/'data/datasets.json'):
   name=v['name'];p=base/v['path'];digest=hashlib.sha256();counts=collections.Counter();index={}
   compressed=not p.exists()
   if compressed:p=p.with_suffix(p.suffix+'.gz')
   with (gzip.open(p,'rb') if compressed else p.open('rb')) as f:
    for i,line in enumerate(f):
     digest.update(line);r=json.loads(line);ex=r['example_id'];counts[len(r['operands'])]+=1
     if ex in wanted:
      assert ex not in self.data or self.data[ex]==r
      self.data[ex]=r;self.members[name].add(ex);index[ex]=i
      self.rowhash[ex]=hashlib.sha256(json.dumps(r,separators=(',',':')).encode()).hexdigest()
   assert digest.hexdigest()==v['sha256'],p
   self.indices[name]=index;self.hist[name]=counts
  assert wanted<=self.data.keys()
  self.scores={}
  for (m,u,d,sp,cohort),qs in self.ps.items():
   assert len({q['example_id'] for q in qs})==len(qs),(m,u,d,sp,cohort)
   for q in qs:
    r=self.data[q['example_id']];assert q['source_index']==self.indices[sp][q['example_id']]
    assert q['row_sha256']==self.rowhash[q['example_id']]
    q['n']=len(r['operands']);q['w']=len(str(abs(r['result'])))
    q['strict']=q['ops']==[x for x,k in zip(r['target_operator_ids'],r['target_mask']) if k]
    try:q['semantic']=execute(r,q['ops'])==r['result']
    except (ValueError,IndexError):q['semantic']=False
   subsets={sp:qs}
   if sp=='fresh_id':subsets['untouched_id']=[q for q in qs if q['example_id'] in self.members['untouched_id']]
   if sp.endswith('train'):
    ds=Path(self.config[m]['data']['train']['path']).stem
    subsets[sp+'_seen']=[q for q in qs if q['example_id'] in self.members[ds]]
    subsets[sp+'_unseen']=[q for q in qs if q['example_id'] not in self.members[ds]]
   for split,sub in subsets.items():
    key=m,u,d,split
    if key in self.scores:assert self.aggregate(self.scores[key])==self.aggregate(sub),('cohort mismatch',key)
    else:self.scores[key]=sub
  self.live={}
  for m in self.config:
   groups=collections.defaultdict(list);last=0;A=self.config[m]['training']['act_steps'];T=self.evals[m][-1]['update']
   for row in lines(base/'records'/m/'curve/live.jsonl.gz'):
    u=row['update'];assert u==last+1;last=u;assert row['act_step']==(u-1)%A+1
    if row['act_step']==A:groups[min(((u-1)//8192+1)*8192,T)].append(row)
   assert last==T
   for end,rs in groups.items():
    k=sum(r['strict_operator_correct'] for r in rs);n=sum(r['total'] for r in rs)
    self.live[m,end]=dict(window_start_update=((end-1)//8192)*8192+1,window_end_update=end,act_step=A,selected_metric_rows=len(rs),first_selected_line=rs[0]['update'],last_selected_line=rs[-1]['update'],correct=k,total=n,percent=100*k/n)
  self.ambiguity={}
  for r in lines(base/'data/id_dev.jsonl'):
   stack=[];i=0
   for marker in r['structure']:
    if marker=='n':stack.append({r['operands'][i]:1});i+=1
    else:
     right,left=stack.pop(),stack.pop();out={}
     for a,ca in left.items():
      for b,cb in right.items():
       for v in [a+b,a-b,a*b]+([a//b] if b and a%b==0 else []):out[v]=min(2,out.get(v,0)+ca*cb)
     stack.append(out)
   n=len(r['operands']);v=self.ambiguity.setdefault(n,dict(total=0,ambiguous_rows=0));v['total']+=1;v['ambiguous_rows']+=stack[0].get(r['result'],0)>1
  self.charges={m:js(base/'records'/m/'terminal.json')['charged_seconds'] for m in self.config}
  for m,i in [('R524',5),('R524-B4096-D16-short',6),('R262',7)]:self.charges[m]+=js(base/'records/allocations'/f'{i:02d}_terminal.json')['charged_seconds']
 def aggregate(self,qs):return dict(total=len(qs),strict=sum(q['strict'] for q in qs),semantic=sum(q['semantic'] for q in qs))
 def model_update(self,m):
  if '_u' in m:a,u=m.rsplit('_u',1);return a,int(u)
  if m.endswith('_best'):a=m.removesuffix('_best');return a,self.best[a]
  a=m.removesuffix('_final');return a,self.evals[a][-1]['update']
 def score(self,m,d,sp,group='overall',n=None,w=None):
  m,u=self.model_update(m);qs=self.scores[m,u,int(d),sp]
  if group in ['operand_count','joint','operand_count_and_result_width']:qs=[q for q in qs if q['n']==int(n)]
  if group in ['result_width','joint','operand_count_and_result_width']:qs=[q for q in qs if q['w']==int(w)]
  return self.aggregate(qs)
 def curve(self,m,d,u):
  e=next(e for e in self.evals[m] if e['update']==int(u));return e.get('ema_id_'+str(d),e)
 def endpoint(self,m,d,which='final'):
  es=self.evals[m]
  if which=='final':e=es[-1]
  else:
   field='semantic_correct' if which=='semantic' else 'strict_correct'
   e=max(es,key=lambda q:q.get('ema_id_'+str(d),q)[field])
  return e['update'],e.get('ema_id_'+str(d),e)
 def standard(self,m,d,uncertainty=False):
  dev=self.score(m,d,'id_dev');fresh=self.score(m,d,'fresh_id');D,Q,N,K=dev['total'],dev['strict'],fresh['total'],fresh['strict']
  ds={n:self.score(m,d,'id_dev','operand_count',n) for n in range(3,9)};fs={n:self.score(m,d,'fresh_id','operand_count',n) for n in range(3,9)}
  weights={n:ds[n]['total']/D for n in ds};p=sum(weights[n]*fs[n]['strict']/fs[n]['total'] for n in ds)
  deficit=100*(Q/D-p) if uncertainty else 100*Q/D-100*p
  vf=math.fsum(weights[n]**2*(fs[n]['strict']/fs[n]['total'])*(1-fs[n]['strict']/fs[n]['total'])/fs[n]['total'] for n in ds);se=100*math.sqrt((Q/D)*(1-Q/D)/D+vf)
  return dict(development_correct=Q,development_total=D,fresh_correct=K,fresh_total=N,development_percent=100*Q/D,fresh_percent=100*K/N,raw_deficit_pp=100*(Q/D-K/N),standardized_fresh_percent=100*p,standardized_deficit_pp=deficit,approximate_SE_pp=se,z=deficit/se,fresh_ood_percent=100*self.score(m,d,'fresh_ood')['strict']/N)
 def fill(self,name,row):
  r=dict(row);m=r.get('model','');d=int(r.get('depth') or self.primary.get(m.removesuffix('_final').removesuffix('_best'),8))
  def put(**kw):r.update({k:str(v) for k,v in kw.items() if k in r})
  if name in ['recorded_curves','recorded_curves_recomputed','Figure_3_data','Figure_S1_data']:
   e=self.curve(m,d,r['update']);put(strict=e['strict_correct'],semantic=e['semantic_correct'],total=e['total'])
  elif name=='recorded_endpoints':
   u,e=self.endpoint(m,d);bu,b=self.endpoint(m,d,'best');su,s=self.endpoint(m,d,'semantic')
   put(primary=d==self.primary[m],final_update=u,final_strict=e['strict_correct'],final_semantic=e['semantic_correct'],best_update=bu,best_strict=b['strict_correct'],semantic_at_best_strict=b['semantic_correct'],best_semantic=s['semantic_correct'],best_semantic_update=su,evaluations=len(self.evals[m]),total=e['total'])
  elif name=='recorded_endpoint_strata':
   u,e=self.endpoint(m,d,r['selection']);i,s=next((i,s) for i,s in enumerate(e['strata']) if s['operand_count']==int(r['operand_count']) and s['result_width']==int(r['result_width']))
   put(update=u,total=s['total'],strict_correct=s['strict_correct'],semantic_correct=s['semantic_correct'],source_line=next(i+1 for i,e in enumerate(self.evals[m]) if e['update']==u),source_stratum=i)
  elif name.startswith('cpu_'):
   sp=r['split'];g=r.get('group','operand_count' if name=='cpu_operand_profiles' else 'overall')
   v=self.score(m,d,sp,g,r.get('operand_count'),r.get('result_width'))
   put(**v,strict_correct=v['strict'],semantic_correct=v['semantic'])
   if name=='cpu_intervals':
    k,n=v[r['metric']],v['total'];lo,hi=wilson(k,n);put(correct=k,accuracy=k/n,wilson_low=lo,wilson_high=hi)
  elif name.startswith('membership_'):
   model=m+'_'+r['selection'];sp=r['source_sample'];g='operand_count' if name=='membership_by_operand' else 'overall';n=r.get('operands')
   a=self.score(model,8,sp+'_seen',g,n);b=self.score(model,8,sp+'_unseen',g,n)
   put(seen_n=a['total'],seen_strict=a['strict'],seen_semantic=a['semantic'],unseen_n=b['total'],unseen_strict=b['strict'],unseen_semantic=b['semantic'],strict_gap_pp=100*(a['strict']/a['total']-b['strict']/b['total']) if a['total'] and b['total'] else '')
  elif name in ['fresh_id_standardization','standardization_uncertainty']:put(**self.standard(m+'_final',d,name=='standardization_uncertainty'))
  elif name=='shared_untouched_profiles':
   n=int(r['operand_count']);a=self.score(m+'_final',self.primary[m],'fresh_id','operand_count',n);b=self.score(m+'_final',self.primary[m],'untouched_id','operand_count',n)
   k,N=a['strict']-b['strict'],a['total']-b['total'];put(shared_correct=k,shared_total=N,shared_percent=100*k/N,untouched_correct=b['strict'],untouched_total=b['total'],untouched_percent=100*b['strict']/b['total'])
  elif name=='live_training_windows':put(**self.live[m,int(r['window_end_update'])])
  elif name=='live_development_readouts':
   u,e=self.endpoint(m,d);a=self.live[m,u];dp=100*e['strict_correct']/e['total'];put(start=a['window_start_update'],end=u,live_correct=a['correct'],live_total=a['total'],live_percent=a['percent'],development_correct=e['strict_correct'],development_total=e['total'],development_percent=dp,gap_pp=a['percent']-dp)
  elif name=='label_ambiguity':
   a=self.ambiguity[int(r['operand_count'])];put(**a,ambiguity_percent=100*a['ambiguous_rows']/a['total'])
  elif name=='inherited_failed_charges':
   i=int(r['record'][1:]);v=js(self.base/'records/allocations'/f'{i:02d}_terminal.json')['charged_seconds'];prev=js(self.base/'records/allocations'/f'{i-1:02d}_terminal.json')['charged_seconds'] if i>1 else 0;put(charged_seconds=v-prev)
  elif name=='optimization_clocks':
   t=self.config[m]['training'];T=t['total_updates'];b=t.get('effective_batch',t['physical_batch']);a=t['act_steps'];w=t['warmup_updates'];peak=t['peak_lr'];low=t['min_lr_ratio']
   def lr(u):return peak*u/w if u<=w else peak*(low+(1-low)*.5*(1+math.cos(math.pi*(u-w)/(T-w))))
   put(updates=T,batch=b,held_updates=a,warmup_updates=w,warmup_schedule_percent=100*w/T,cosine_progress_at_38912_percent=100*(38912-w)/(T-w),learning_rate_at_38912=lr(38912),sum_learning_rate=math.fsum(lr(u) for u in range(1,T+1)),warmup_fresh_presentations=w*b//a,ema_half_life_fresh_rows=math.log(.5)/math.log(t['ema_decay'])*b/a)
  elif name=='checkpoint_inventory':
   u,e=self.endpoint(m,self.primary[m],r['kind']);c=next(c for c in self.checkpoints[m] if c['update']==u);put(update=u,sha256=c['sha256'])
  elif name=='fixture_checks':
   a=js(self.base/'records/fixtures/F01_FP64_EQUIVALENCE.json');maxes=a['max_abs_error'];check=r['check']
   if check.startswith('Accumulation:'):
    fs={'Accumulation: parameters and EMA':['parameters','ema'],'Accumulation: optimizer and loss':['optimizer','loss'],'Accumulation: carry':['carry']}
    v=max(x['gradient_max_abs_error'] for x in a['synchronized_checks']) if 'synchronized' in check else max(v[f] for v in maxes.values() for f in fs[check]);put(maximum_discrepancy=f'{v:.16g}')
  elif name=='Figure_1_data':
   sp=r['stream'];put(training_rows=self.config[m]['data']['train']['count'])
   if sp in ['original_train_seen','original_train_unseen']:v=self.score(m+'_final',8,sp);k,n=v['strict'],v['total']
   else:u,e=self.endpoint(m,self.primary[m]);k,n=e['strict_correct'],e['total']
   put(correct=k,total=n,percent=100*k/n)
  elif name=='Figure_2_data':
   u,a=self.endpoint(m,self.primary[m]);v,b=self.endpoint(m,self.primary[m],'best');n=a['total'];put(final_count=a['strict_correct'],best_count=b['strict_correct'],total=n,final_percent=100*a['strict_correct']/n,best_percent=100*b['strict_correct']/n,charged_GPU_hours=self.charges[m]/3600)
  elif name=='Figure_S2_data':
   u,e=self.endpoint(m,self.primary[m]);ss=[s for s in e['strata'] if s['operand_count']==int(r['operand_count'])];put(total=sum(s['total'] for s in ss),strict=sum(s['strict_correct'] for s in ss),semantic=sum(s['semantic_correct'] for s in ss))
  elif name=='Figure_S3_data':
   a=self.score(m+'_final',self.primary[m],'id_dev');b=self.score(m+'_final',self.primary[m],'fresh_id');put(dev_correct=a['strict'],dev_total=a['total'],fresh_correct=b['strict'],fresh_total=b['total'],dev_percent=100*a['strict']/a['total'],fresh_percent=100*b['strict']/b['total'])
  elif name=='Figure_S4_data':put(**self.score(m,d,'fresh_ood','operand_count',r['operand_count']))
  elif name=='Figure_S5_data':
   u=int(r['update']);m='R65-original'
   if r['stream']=='live_final_step':put(**self.live[m,u])
   else:e=self.curve(m,8,u);put(correct=e['strict_correct'],total=e['total'],percent=100*e['strict_correct']/e['total'])
  else:raise ValueError(name)
  return r

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',type=Path,default=BASE);args=ap.parse_args();ev=Evidence(args.base);args.output.mkdir(parents=True,exist_ok=True)
 plans=js(args.base/'records/table_layouts.json');cells=0
 for name,plan in plans.items():
  rs=[ev.fill(name,row) for row in plan['rows']]
  with (args.output/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=plan['fields']);w.writeheader();w.writerows(rs)
  cells+=len(rs)*len(plan['fields'])
 from supplement import build
 build(ev,args.output)
 from main_tables import build as main_tables
 main_tables(ev,args.output)
 print(json.dumps(dict(tables=len(list(args.output.glob("*.csv"))),source_table_cells=cells,prediction_groups=len(ev.ps),recomputed_score_sets=len(ev.scores),live_windows=len(ev.live))))
if __name__=='__main__':main()
