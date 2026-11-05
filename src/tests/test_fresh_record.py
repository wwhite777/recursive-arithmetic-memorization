"""Exercise actual fresh sampling, compact verification, corruption and resume."""
import copy
import hashlib
import json
import struct
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from qdriver.engine import Engine
from qdriver.fresh_record import verify_record
from qdriver.workload import config_for_schedule, validate_config
from .helpers import tiny_config, rows, run_root, exact


class FreshRecordTests(unittest.TestCase):
    def engine(self, name, total=12):
        config=tiny_config(1,total)
        config.update(model_mode='nonrecursive',fresh_batch=True)
        return Engine(config,*rows(),run_root()/name,'7'*64)

    def test_every_update_draws_and_commitments_regenerate(self):
        engine=self.engine('fresh-sequence'); generator=torch.Generator().manual_seed(101)
        expected=[]
        for epoch in range(3):
            order=torch.randperm(8,generator=generator)
            for i in range(0,8,2):expected.append(order[i:i+2].tolist())
        actual=[];original=engine.sampler.batch
        def observe(*args):
            batch=original(*args);actual.append(engine.sampler.indices.tolist());return batch
        with patch.object(engine.sampler,'batch',side_effect=observe), patch.object(engine.trainer.model,'forward',wraps=engine.trainer.model.forward) as forward:
            for update,indices in enumerate(expected):
                engine.step()
                self.assertEqual(actual[-1],indices)
                self.assertEqual((engine.trainer.updates,engine.trainer.optimizer_steps,engine.trainer.ema_steps),(update+1,)*3)
            self.assertEqual(forward.call_count,12)
        result=verify_record(engine.output)
        self.assertEqual((result['verified_updates'],result['verified_epochs'],result['complete']),(12,3,True))
        saved=engine.journal.rows('samples')
        for r,indices in zip(saved,expected):
            self.assertEqual(r['indices_sha256'],hashlib.sha256(struct.pack('<QQ',*indices)).hexdigest())

    def test_first_update_matches_held_single_pass(self):
        fresh=self.engine('fresh-first',total=16)
        config=tiny_config(8,16);config['model_mode']='nonrecursive'
        held=Engine(config,*rows(),run_root()/'held-first','7'*64)
        self.assertEqual(fresh.step(),held.step())
        for key in ('model','optimizer','ema','updates','optimizer_steps','ema_steps'):
            exact(fresh.trainer.state_dict()[key],held.trainer.state_dict()[key])

    def test_wrong_batch_and_malformed_records_fail(self):
        engine=self.engine('fresh-corrupt')
        for _ in range(12):engine.step()
        samples=engine.output/'samples.jsonl';original=samples.read_bytes()
        records=[json.loads(x) for x in original.splitlines()]
        records[5]['indices_sha256']=hashlib.sha256(struct.pack('<QQ',0,0)).hexdigest()
        samples.write_text(''.join(json.dumps(x)+'\n' for x in records))
        with self.assertRaisesRegex(ValueError,'batch 5 mismatch'):verify_record(engine.output)
        samples.write_bytes(original)
        for name in ('samples.jsonl','epochs.jsonl'):
            p=engine.output/name;before=p.read_bytes()
            for bad in (b'',before[:-1],before+before.splitlines(keepends=True)[0]):
                p.write_bytes(bad)
                with self.assertRaises(ValueError):verify_record(engine.output)
            p.write_bytes(before)
        p=engine.output/'FRESH_RECORD.json';meta=json.loads(p.read_text());meta['torch_version']='wrong';p.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Torch version'):verify_record(engine.output)

    def test_resume_across_epoch_is_exact(self):
        a=self.engine('fresh-resume');b=self.engine('fresh-reference')
        for _ in range(3):a.step()
        checkpoint=a.checkpoint()
        for _ in range(3):a.step()
        resumed=self.engine('fresh-resume');resumed.resume(checkpoint)
        for _ in range(9):resumed.step()
        for _ in range(12):b.step()
        exact(resumed.trainer.state_dict(),b.trainer.state_dict())
        self.assertTrue(verify_record(resumed.output)['complete'])
        self.assertEqual((resumed.output/'samples.jsonl').read_bytes(),(b.output/'samples.jsonl').read_bytes())
        self.assertTrue(list((resumed.output/'discarded_tails').glob('epochs.*.jsonl')))

    def test_partial_record_requires_explicit_count(self):
        engine=self.engine('fresh-partial')
        for _ in range(5):engine.step()
        with self.assertRaises(ValueError):verify_record(engine.output)
        self.assertFalse(verify_record(engine.output,5)['complete'])

    def test_actual_configs_validate_and_changes_are_rejected(self):
        base=Path(__file__).resolve().parents[2]
        for model in ('S65f','S524f'):
            config=json.loads((base/'configs'/(model+'.json')).read_text());validate_config(config)
            schedule=json.loads((base/'configs'/(model+'_schedule.json')).read_text())
            derived=config_for_schedule(schedule,training_data=config['data']['train'])
            validate_config(derived);self.assertEqual(derived['training'],config['training'])
            for key,value in (('seed',202),('model_mode','recursive'),('fresh_batch',1)):
                bad=copy.deepcopy(config);bad[key]=value
                with self.assertRaises(ValueError):validate_config(bad)
            for key,value in (('act_steps',8),('total_updates',2048),('peak_lr',.001)):
                bad=copy.deepcopy(config);bad['training'][key]=value
                with self.assertRaises(ValueError):validate_config(bad)
