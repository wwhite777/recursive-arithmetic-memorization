"""Release configuration construction and hash-checked training data."""
import copy
from pathlib import Path
from .common import FROZEN, PROJECT, read_json, require, sha

def config_for(batch, steps, total, *, effective_batch=None, training_data=None, seed=101, model='recursive'):
    config=copy.deepcopy(read_json(FROZEN))
    config['seed']=config['data_order_seed']=seed
    config['training'].update(physical_batch=batch,act_steps=steps,total_updates=total,seed=seed)
    config['schema']='accumulated-config-v1' if effective_batch is not None else 'standard-config-v1'
    if effective_batch is not None:config['training']['effective_batch']=effective_batch
    if training_data is not None:config['data']['train']=dict(training_data)
    if model=='nonrecursive':config['model_mode']=model
    return config

def validate_config(config):
    require(config['model']==read_json(FROZEN)['model'],'model configuration differs')
    require(config['training']['act_steps'] in (8,16),'held-batch count')
    require(config['seed']==config['data_order_seed']==config['training']['seed'],'seed streams')

def config_for_schedule(schedule, *, training_data=None, seed=101, model='recursive'):
    return config_for(schedule['batch'],schedule['act_steps'],schedule['T'],effective_batch=schedule.get('effective_batch'),training_data=training_data,seed=seed,model=model)

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
