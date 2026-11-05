"""Full-state resumable training and crash-consistent journal prefixes."""
import hashlib
import os
from pathlib import Path
import random
import time
import uuid

from .common import INTERVAL, atomic_bytes, canonical, fsync_dir, identity, object_sha, owned, read_json, require, save_json, sha
from .training import AccumulationModel, ArithmeticModel, NonrecursiveModel, PilotConfig, Trainer, TrainingConfig, evaluate, tensors, torch
from .rules import StopRules
from .fresh_record import batch_record, epoch_record, record_metadata


class Journal:
    NAMES = ('metrics', 'samples', 'evaluations')

    def __init__(self, root, fresh_batch=False):
        if fresh_batch:
            self.NAMES = (*self.NAMES, 'epochs')
        self.root = owned(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.NAMES:
            path = self.root/(name+'.jsonl')
            if not path.exists():
                atomic_bytes(path, b'', new=True)

    def append(self, name, value):
        with (self.root/(name+'.jsonl')).open('ab') as stream:
            stream.write(canonical(value)+b'\n')
            stream.flush()

    def snapshot(self):
        state = {}
        for name in self.NAMES:
            p = self.root/(name+'.jsonl')
            with p.open('rb') as stream:
                os.fsync(stream.fileno())
            state[name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        return state

    def restore(self, state):
        require(set(state) == set(self.NAMES), 'checkpoint log set')
        # Validate every prefix before modifying any log. Preserve every suffix.
        for name, spec in state.items():
            p = self.root/(name+'.jsonl')
            require(sha(p, spec['bytes']) == spec['sha256'], 'checkpoint log identity changed: '+name)
        if 'epochs' in self.NAMES:
            # Enforce the replay-tail premise used by the fresh projection,
            # including explicit attempts to rewind farther than one interval.
            meta = read_json(self.root/'FRESH_RECORD.json')
            batches_per_epoch = meta['length']//meta['batch_size']
            limits = dict(metrics=INTERVAL, samples=INTERVAL, evaluations=1,
                          epochs=(INTERVAL+batches_per_epoch-1)//batches_per_epoch+1)
            for name, spec in state.items():
                with (self.root/(name+'.jsonl')).open('rb') as stream:
                    stream.seek(spec['bytes'])
                    for count, _ in enumerate(stream, 1):
                        require(count <= limits[name], 'fresh replay tail exceeds reserved interval: '+name)
        for name, spec in state.items():
            p = self.root/(name+'.jsonl')
            if p.stat().st_size > spec['bytes']:
                with p.open('rb') as stream:
                    stream.seek(spec['bytes'])
                    tail = stream.read()
                atomic_bytes(self.root/'discarded_tails'/(name+'.'+uuid.uuid4().hex+'.jsonl'), tail, new=True)
                with p.open('r+b') as stream:
                    stream.truncate(spec['bytes'])
                    stream.flush()
                    os.fsync(stream.fileno())

    def rows(self, name):
        import json
        return [json.loads(x) for x in (self.root/(name+'.jsonl')).read_text().splitlines()]


class Sampler:
    def __init__(self, train, batch_size, steps, device, seed=101, fresh_batch=False):
        require(not fresh_batch or steps == 1, 'compact record requires fresh batches')
        self.fresh_batch = fresh_batch
        self.train, self.batch_size, self.steps, self.device = train, batch_size, steps, device
        self.length = len(train['token_ids'])
        require(self.length >= batch_size and self.length % batch_size == 0, 'full physical training batches required')
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.order, self.cursor, self.epoch, self.indices, self.current = None, self.length, -1, None, None

    def batch(self, update, journal):
        if update % self.steps == 0:
            if self.cursor >= self.length:
                if self.fresh_batch:
                    generator_state = self.generator.get_state()
                self.order = torch.randperm(self.length, generator=self.generator)
                self.cursor = 0
                self.epoch += 1
                if self.fresh_batch:
                    journal.append('epochs', epoch_record(self.epoch, update, self.order, generator_state))
            self.indices = self.order[self.cursor:self.cursor+self.batch_size].clone()
            self.cursor += self.batch_size
            self.current = {k: v.index_select(0, self.indices).to(self.device) for k, v in self.train.items()}
            if self.fresh_batch:
                journal.append('samples', batch_record(update, self.epoch, self.indices))
            else:
                journal.append('samples', {'trajectory': update//self.steps, 'epoch': self.epoch, 'indices': self.indices.tolist()})
        require(self.current is not None, 'missing held batch')
        return self.current

    def state_dict(self):
        return {'generator_state': self.generator.get_state(), 'order': self.order, 'cursor': self.cursor,
                'epoch': self.epoch, 'indices': self.indices, 'length': self.length, 'batch_size': self.batch_size, 'steps': self.steps}

    def load(self, state):
        require((state['length'], state['batch_size'], state['steps']) == (self.length, self.batch_size, self.steps), 'sampler shape changed')
        self.generator.set_state(state['generator_state'])
        self.order, self.indices = state['order'], None
        self.cursor, self.epoch = state['cursor'], state['epoch']
        require(0 <= self.cursor <= self.length and self.cursor % self.batch_size == 0, 'sampler cursor')
        if self.order is not None:
            require(torch.equal(self.order.sort().values, torch.arange(self.length)), 'sampler order not permutation')
        self.current = None


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(x, device) for x in value)
    if isinstance(value, list):
        return [to_device(x, device) for x in value]
    return value


class Engine:
    def __init__(self, config, train_rows, id_rows, output, manifest_sha, device='cpu'):
        self.config, self.device = config, torch.device(device)
        self.output = owned(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.identities = identity(config, manifest_sha)
        require(config['seed'] == config['data_order_seed'] == config['training']['seed'], 'seed streams disagree')
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        model_class = AccumulationModel if config['training'].get('effective_batch') is not None else ArithmeticModel
        if config.get('model_mode') == 'nonrecursive':
            require(config['training'].get('effective_batch') is None
                    and config['training']['act_steps'] == (1 if config.get('fresh_batch') else 8),
                    'NR requires held1 with fresh_batch, otherwise held8 training')
            model_class = NonrecursiveModel
        model = model_class(PilotConfig(**config['model']))
        if model.config.forward_dtype == 'float64':
            model = model.double()
        model = model.to(self.device)
        self.trainer = Trainer(model, TrainingConfig(**config['training']), 'S22')
        self.sampler = Sampler(tensors(train_rows), self.trainer.config.effective_batch or self.trainer.config.physical_batch,
                               self.trainer.config.act_steps, self.device, seed=config['data_order_seed'],
                               fresh_batch=config.get('fresh_batch', False))
        self.id_rows = id_rows
        self.journal = Journal(self.output, fresh_batch=config.get('fresh_batch', False))
        if config.get('fresh_batch'):
            meta = record_metadata(config['data_order_seed'], self.sampler.length, self.sampler.batch_size,
                                   self.trainer.config.total_updates, self.identities, torch.__version__)
            path = self.output/'FRESH_RECORD.json'
            if path.exists():
                require(read_json(path) == meta, 'fresh record metadata changed')
            else:
                save_json(path, meta, new=True)
        self.evaluations = []
        self.compute_seconds = self.evaluation_seconds = self.checkpoint_seconds = 0.0

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def step(self):
        t = self.trainer
        batch = self.sampler.batch(t.updates, self.journal)
        collect = t.updates == 0 or (t.updates+1) % 100 == 0
        self.sync()
        started = time.monotonic()
        item = t.step(batch, collect_displacement=collect)
        self.sync()
        self.compute_seconds += time.monotonic()-started
        require(item['update'] == t.optimizer_steps == t.ema_steps, 'trainer clock drift')
        self.journal.append('metrics', item)
        if t.updates % t.config.act_steps == 0:
            # The next step draws a new batch/plan. Nothing from this held batch
            # participates in it, so release it before evaluation/checkpointing.
            t.carry = t.masks = t.current_batch_identity = None
            self.sampler.current = self.sampler.indices = None
        return item

    def evaluate(self):
        self.sync()
        started = time.monotonic()
        result = dict(evaluate(self.trainer, self.id_rows, self.device))
        if self.config['schema'] == 'experiment':
            # Top-level experiment fields remain the training-depth primary used by all
            # stop/FLAT/CAP paths. Both depths start from fresh EMA carry.
            primary = dict(result)
            primary_steps = self.trainer.config.act_steps
            secondary_steps = 8 if primary_steps == 16 else 16
            secondary = dict(evaluate(self.trainer, self.id_rows, self.device, act_steps=secondary_steps))
            require(primary['total'] == secondary['total'], 'dual-depth row count changed')
            result.update(primary_steps=primary_steps, secondary_steps=secondary_steps,
                          **{f'ema_id_{primary_steps}': primary, f'ema_id_{secondary_steps}': secondary})
        elif self.config.get('model_mode') == 'nonrecursive':
            result['primary_steps'] = 1
        self.sync()
        self.evaluation_seconds += time.monotonic()-started
        result['update'] = self.trainer.updates
        if self.config['schema'] == 'experiment' or self.config.get('fixed_T', False):
            flat = None
            # Short calibration probes also time evaluation, off main cadence.
            if result['update'] > 0 and result['update'] % INTERVAL == 0:
                rule = StopRules([{'update': x['update'], 'strict_correct': x['strict_correct'], 'total': x['total']}
                                  for x in self.evaluations], stop_on_flat=False)
                rule.observe(result['update'], result['strict_correct'], result['total'])
                flat = rule.flat_record()
            result.update(flat_record=flat, flat_stops=False)
        self.evaluations.append(result)
        self.journal.append('evaluations', result)
        return result

    def checkpoint(self):
        require(self.trainer.updates % self.trainer.config.act_steps == 0,
                'checkpoint requires a held-batch boundary')
        require(self.trainer.carry is None and self.trainer.masks is None
                and self.trainer.current_batch_identity is None
                and self.sampler.current is None and self.sampler.indices is None
                and getattr(self.trainer.model, 'step_noise', None) is None,
                'checkpoint has live held-batch state')
        self.sync()
        started = time.monotonic()
        previous = read_json(self.output/'LATEST.json') if (self.output/'LATEST.json').exists() else None
        trainer = self.trainer.state_dict()
        for key in ('carry', 'masks', 'current_batch_identity'):
            del trainer[key]
        sampler = self.sampler.state_dict()
        del sampler['indices']
        # Retain the existing gradient payload; the next update zeroes it.
        gradients = {n: None if p.grad is None else p.grad.detach().clone() for n, p in self.trainer.model.named_parameters()}
        state = {'schema': 'experiment', 'identities': self.identities, 'trainer': trainer,
                 'gradients': gradients, 'model_training': self.trainer.model.training,
                 'sampler': sampler, 'evaluations': self.evaluations,
                 'clocks': {'updates': self.trainer.updates, 'optimizer_steps': self.trainer.optimizer_steps,
                            'ema_steps': self.trainer.ema_steps, 'act_step': self.trainer.updates % self.trainer.config.act_steps,
                            'trajectory': self.trainer.updates // self.trainer.config.act_steps,
                            'compute_seconds': self.compute_seconds, 'evaluation_seconds': self.evaluation_seconds,
                            'checkpoint_seconds_before_this_write': self.checkpoint_seconds},
                 'rng': {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
                         'python': random.getstate()}, 'logs': self.journal.snapshot()}
        directory = self.output/'checkpoints'
        directory.mkdir(exist_ok=True)
        path = directory/f'u{self.trainer.updates:09d}-{uuid.uuid4().hex}.pt'
        temp = path.with_suffix('.pt.tmp')
        with temp.open('xb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fsync_dir(directory)
        elapsed = time.monotonic()-started
        self.checkpoint_seconds += elapsed
        ref = {'path': str(path.relative_to(self.output)), 'sha256': sha(path), 'identities': self.identities,
               'update': self.trainer.updates, 'checkpoint_seconds': elapsed}
        self.read_checkpoint(ref)  # deserialize and verify before publication/deletion
        # Immutable generation committed before the atomic pointer. Old valid
        # generations survive a crash or a later interrupted serialization.
        save_json(path.with_suffix('.json'), ref, new=True)
        self.retain_best_ema(ref, self.evaluations)
        save_json(self.output/'LATEST.json', ref)
        index_path = self.output/'CHECKPOINTS.json'
        records = read_json(index_path)['verified'] if index_path.exists() else []
        if previous is not None and not any(x['path'] == previous['path'] for x in records):
            # Recover publication interrupted between LATEST and its index.
            self.read_checkpoint(previous)
            records.append(previous)
        records.append(ref)
        save_json(index_path, {'verified': records})
        self.prune_checkpoints(records)
        return ref

    def retain_best_ema(self, ref, evaluations):
        """Pin an evaluated generation before pruning; earliest strict maximum wins."""
        if not evaluations or evaluations[-1]['update'] != ref['update']:
            return
        result = evaluations[-1]
        path = self.output/'BEST_EMA.json'
        previous = read_json(path) if path.exists() else None
        if previous is not None:
            require(previous['checkpoint']['identities'] == self.identities, 'best EMA checkpoint identity mismatch')
            require(owned(self.output/previous['checkpoint']['path']).is_file(), 'best EMA checkpoint missing')
            require(previous['total'] == result['total'], 'best EMA row count changed')
            if result['strict_correct'] <= previous['strict_correct']:
                return
        save_json(path, {'checkpoint': ref, 'update': ref['update'], 'strict_correct': result['strict_correct'],
                        'total': result['total'], 'primary_steps': result.get('primary_steps', self.trainer.config.act_steps),
                        'selection': 'earliest checkpoint at maximum primary EMA-ID strict_correct'})

    def prune_checkpoints(self, records):
        # Publication and the verified-generation record precede every unlink.
        protected = {read_json(self.output/'LATEST.json')['path']}
        resume_path = self.output/'RESUME.json'
        if resume_path.exists():
            protected.add(read_json(resume_path)['checkpoint']['path'])
        best_path = self.output/'BEST_EMA.json'
        if best_path.exists():
            protected.add(read_json(best_path)['checkpoint']['path'])
        keep = protected | {x['path'] for x in records[-2:]}
        for ref in records[:-2]:
            if ref['path'] in keep:
                continue
            path = owned(self.output/ref['path'])
            require(path.parent == self.output/'checkpoints', 'prune only checkpoint generations')
            path.unlink(missing_ok=True)
            path.with_suffix('.json').unlink(missing_ok=True)
        fsync_dir(self.output/'checkpoints')

    def read_checkpoint(self, ref):
        require(ref['identities'] == self.identities, 'checkpoint manifest/config/source identity mismatch')
        path = owned(self.output/ref['path'])
        require(path.is_relative_to(self.output/'checkpoints') and path.suffix == '.pt', 'only own full-state checkpoint')
        require(sha(path) == ref['sha256'], 'checkpoint digest mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        require(state['schema'] == 'experiment' and state['identities'] == self.identities, 'checkpoint schema/identity')
        require(state['trainer']['updates'] == ref['update'] == state['clocks']['updates'], 'checkpoint update identity')
        for key in ('updates', 'optimizer_steps', 'ema_steps'):
            require(state['trainer'][key] == ref['update'], 'checkpoint clocks differ')
        require(state['clocks']['act_step'] == ref['update'] % self.trainer.config.act_steps, 'ACT clock mismatch')
        require(state['clocks']['act_step'] == 0, 'checkpoint requires a held-batch boundary')
        require(not {'carry', 'masks', 'current_batch_identity'} & state['trainer'].keys()
                and 'indices' not in state['sampler'], 'checkpoint contains held-batch state')
        require(set(state.get('rng', {})) == {'cpu', 'cuda', 'python'}, 'checkpoint RNG state incomplete')
        if self.device.type == 'cuda':
            require(len(state['rng']['cuda']) == torch.cuda.device_count() == 1, 'CUDA RNG/device count mismatch')
        else:
            require(not state['rng']['cuda'], 'CPU fixture cannot load GPU state')
        return state

    def resume(self, ref=None):
        ref = read_json(self.output/'LATEST.json') if ref is None else ref
        state = self.read_checkpoint(ref)
        self.retain_best_ema(ref, state['evaluations'])
        self.journal.restore(state['logs'])
        trainer = dict(state['trainer'], carry=None, masks=None, current_batch_identity=None)
        self.trainer.load_state_dict(to_device(trainer, self.device))
        for n, p in self.trainer.model.named_parameters():
            p.grad = to_device(state['gradients'][n], self.device)
        self.trainer.model.train(state['model_training'])
        self.sampler.load(state['sampler'])
        self.evaluations = state['evaluations']
        self.compute_seconds = state['clocks']['compute_seconds']
        self.evaluation_seconds = state['clocks']['evaluation_seconds']
        self.checkpoint_seconds = state['clocks']['checkpoint_seconds_before_this_write']+ref['checkpoint_seconds']
        torch.set_rng_state(state['rng']['cpu'])
        if self.device.type == 'cuda':
            require(len(state['rng']['cuda']) == torch.cuda.device_count() == 1, 'CUDA RNG/device count mismatch')
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        else:
            require(not state['rng']['cuda'], 'CPU fixture cannot load GPU state')
        random.setstate(state['rng']['python'])
        save_json(self.output/'RESUME.json', {'checkpoint': ref, 'resumed_monotonic': time.monotonic()})
        return ref
