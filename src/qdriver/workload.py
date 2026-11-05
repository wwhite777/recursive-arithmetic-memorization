"""Release configuration construction and hash-checked training data."""
import copy
from pathlib import Path
from .common import FROZEN, PROJECT, read_json, require, sha

def config_for(batch, steps, total, *, effective_batch=None, training_data=None, seed=101, model='recursive', fresh_batch=False):
    config=copy.deepcopy(read_json(FROZEN))
    config['seed']=config['data_order_seed']=seed
    config['training'].update(physical_batch=batch,act_steps=steps,total_updates=total,seed=seed)
    config['schema']='accumulated-config-v1' if effective_batch is not None else 'standard-config-v1'
    if effective_batch is not None:config['training']['effective_batch']=effective_batch
    if training_data is not None:config['data']['train']=dict(training_data)
    if model=='nonrecursive':config['model_mode']=model
    if fresh_batch:config['fresh_batch']=True
    return config

def validate_config(config):
    require(config['model']==read_json(FROZEN)['model'],'model configuration differs')
    fresh=config.get('fresh_batch',False)
    require(type(fresh) is bool,'fresh_batch must be boolean')
    require(config['training']['act_steps'] in ((1,) if fresh else (8,16)),'held-batch count')
    if fresh:
        t=config['training'];base=read_json(FROZEN)
        allowed=[read_json(PROJECT/'configs'/name)['data']['train'] for name in ('S65f.json','S524f.json')]
        require(config.get('model_mode')=='nonrecursive' and config['seed']==101
                and t['total_updates']==196608 and t['physical_batch']==1024
                and t.get('effective_batch') is None and config['data']['train'] in allowed,
                'fresh batch requires S65f/S524f configuration')
        require(t==dict(base['training'],act_steps=1,total_updates=196608,physical_batch=1024,seed=101),
                'fresh optimizer/schedule settings changed')
        require(config['data']['id_dev']==base['data']['id_dev']
                and config['evaluation_batch']==512 and config['evaluation_interval']==config['checkpoint_interval']==2048,
                'fresh data/evaluation settings changed')
    require(config['seed']==config['data_order_seed']==config['training']['seed'],'seed streams')

def config_for_schedule(schedule, *, training_data=None, seed=101, model='recursive', fresh_batch=False):
    steps = schedule['steps'] if 'steps' in schedule else schedule['act_steps']
    return config_for(schedule['batch'],steps,schedule['T'],effective_batch=schedule.get('effective_batch'),training_data=training_data,seed=seed,model=schedule.get('model_mode',model),fresh_batch=schedule.get('fresh_batch',False))

def training_file(path,expected_sha256):
    path=Path(path).resolve()
    require(sha(path)==expected_sha256,'training SHA256 mismatch')
    with path.open() as f:count=sum(bool(x.strip()) for x in f)
    require(count>=1024 and count%1024==0,'full physical batches required')
    return dict(path=str(path),sha256=expected_sha256,count=count)

def load_rows(config):
    from reasoning.arithmetic import load_records
    result={}
    for name,spec in config['data'].items():
        path=Path(spec['path'])
        if not path.is_absolute():path=PROJECT/path
        require(sha(path)==spec['sha256'],'data SHA256 mismatch')
        result[name]=load_records(path)
        require(len(result[name])==spec['count'],'data row count')
    return result
