"""Training entry point. Requires a separately provisioned compatible PyTorch."""
import argparse
from pathlib import Path
from qdriver.common import PROJECT, object_sha, read_json
from qdriver.engine import Engine
from qdriver.workload import load_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');a=p.parse_args()
    config=read_json(a.config);data=load_rows(config)
    output=a.output.resolve();output.relative_to(PROJECT)
    engine=Engine(config,data['train'],data['id_dev'],output,object_sha(config),device=a.device)
    if a.resume:engine.resume()
    while engine.trainer.updates<config['training']['total_updates']:
        engine.step()
        if engine.trainer.updates%config['evaluation_interval']==0:
            engine.evaluate();engine.checkpoint()
    print('Completed updates:',engine.trainer.updates)
if __name__=='__main__':main()
