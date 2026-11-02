"""FP64 full-physical-batch oracle, negative control and resumed accumulated state."""
import copy
from dataclasses import asdict
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from qdriver.common import code_identity, read_json, save_json, sha
from qdriver.training import AccumulationModel, ArithmeticModel, PilotConfig, Trainer, TrainingConfig, tensors
from .c2_helpers import close, compared_state, matched_tiny, max_error
from .helpers import exact, rows, run_root


class AccumulationTests(unittest.TestCase):
    def trainers(self, total=40):
        config = matched_tiny(total=total)
        torch.manual_seed(101)
        model = ArithmeticModel(PilotConfig(**config['model'])).double()
        config_ref = dict(config['training'])
        config_ref.pop('effective_batch')
        reference = Trainer(copy.deepcopy(model), TrainingConfig(**config_ref), 'S22')
        trainers = []
        for k in (1, 2, 4):
            adapted = AccumulationModel(model.config).double()
            adapted.load_state_dict(model.state_dict())
            trainers.append(Trainer(adapted, TrainingConfig(**matched_tiny(k, total)['training']), 'S22'))
        return reference, trainers

    def test_K1_K2_K4_match_physical_FP64_with_stochastic_path(self):
        reference, trainers = self.trainers()
        train, _ = rows()
        batch = tensors(train)
        self.assertGreater(batch['target_mask'].sum(-1).unique().numel(), 1)
        self.assertNotEqual(int(batch['target_mask'][:2].sum())*4, int(batch['target_mask'].sum()))
        initial = copy.deepcopy(reference.model.state_dict())
        maximum = {str(k): {name: 0. for name in ('parameters', 'loss', 'optimizer', 'ema', 'carry')} for k in (1, 2, 4)}
        disagreements = []
        synchronized_errors = []
        for update in range(40):
            # Same new sample ordering at each trajectory boundary.
            if update % 16 == 0:
                order = torch.randperm(8, generator=torch.Generator().manual_seed(101+update))
                batch = {name: value[order] for name, value in batch.items()}
            rng = torch.get_rng_state().clone()
            before = reference.state_dict() if update in (0, 15, 16, 39) else None
            expected = reference.step(batch, collect_displacement=True)
            for k, trainer in zip((1, 2, 4), trainers):
                actual = trainer.step(batch, collect_displacement=True)
                try:
                    close(expected, actual, f'K{k}.metric.u{update+1}', atol=1e-8, rtol=1e-8)
                    state_a, state_b = compared_state(reference), compared_state(trainer)
                    for name in state_a:
                        tolerance = 1e-6 if name == 'carry' else 1e-8 if name == 'optimizer' else 1e-10
                        close(state_a[name], state_b[name], f'K{k}.state.u{update+1}.{name}', atol=tolerance, rtol=tolerance)
                except AssertionError as error:
                    disagreements.append(str(error))
                if before is not None:
                    # Isolate one accumulated update from identical incoming
                    # state, separating summation error from recurrent drift.
                    shadow = Trainer(copy.deepcopy(trainer.model), trainer.config, 'S22')
                    shadow_state = copy.deepcopy(before)
                    shadow_state['training_config'] = asdict(shadow.config)
                    shadow.load_state_dict(shadow_state)
                    shadow_metric = shadow.step(batch, collect_displacement=True)
                    close(expected, shadow_metric, f'synchronized K{k}.metric.u{update+1}')
                    close(compared_state(reference), compared_state(shadow), f'synchronized K{k}.state.u{update+1}')
                    ref_grads = {n: p.grad for n, p in reference.model.named_parameters()}
                    shadow_grads = {n: p.grad for n, p in shadow.model.named_parameters()}
                    close(ref_grads, shadow_grads, f'synchronized K{k}.gradient.u{update+1}')
                    synchronized_errors.append({'K': k, 'update': update+1, 'gradient_max_abs_error': max_error(ref_grads, shadow_grads)})
                exact(reference.masks, trainer.masks)
                exact(rng, torch.get_rng_state())
                maximum[str(k)]['loss'] = max(maximum[str(k)]['loss'], abs(expected['loss']-actual['loss']))
                maximum[str(k)]['parameters'] = max(maximum[str(k)]['parameters'], max(
                    (a-b).abs().max().item() for a, b in zip(reference.model.parameters(), trainer.model.parameters())))
                for name in ('optimizer', 'ema', 'carry'):
                    a = reference.optimizer.state_dict() if name == 'optimizer' else getattr(reference, name)
                    b = trainer.optimizer.state_dict() if name == 'optimizer' else getattr(trainer, name)
                    maximum[str(k)][name] = max(maximum[str(k)][name], max_error(a, b))
                self.assertEqual(trainer.optimizer_steps, update+1)
                self.assertEqual(trainer.ema_steps, update+1)
        self.assertTrue(any(not torch.equal(initial[n], p) for n, p in reference.model.state_dict().items()))
        save_json(run_root()/'FP64_EQUIVALENCE.json', {'updates': 40, 'K': [1, 2, 4], 'effective_batch': 8,
            'ACT_steps': 16, 'max_abs_error': maximum,
            'tolerances': {'40_update_parameters_EMA_and_other_state': {'atol': 1e-10, 'rtol': 1e-10},
                '40_update_optimizer_and_metrics_including_loss': {'atol': 1e-8, 'rtol': 1e-8},
                '40_update_carry': {'atol': 1e-6, 'rtol': 1e-6},
                'synchronized_single_update_all_state_gradient_and_metrics': {'atol': 2e-11, 'rtol': 2e-10}},
            'synchronized_checks': synchronized_errors,
            'checks': ['parameters', 'optimizer moments and clocks', 'EMA', 'loss', 'carry', 'masks', 'global RNG', 'trace', 'strict counts'],
            'oracle': 'unmodified ACT16 full-physical-batch Trainer.step using original ArithmeticModel.prepare',
            'stochastic': 'nonzero experiment dropout/noise; full-shape draws, exact masks; variable target counts'})
        self.assertFalse(disagreements, str(maximum)+'\n'+ ('\n'.join(disagreements[:3])))

    def test_planted_per_micro_clipping_fails_equality(self):
        reference, trainers = self.trainers(total=16)
        batch = tensors(rows()[0])
        expected = reference.step(batch)
        bad = trainers[-1]
        backward = torch.Tensor.backward
        def wrong_backward(value, *args, **kwargs):
            result = backward(value, *args, **kwargs)
            torch.nn.utils.clip_grad_norm_(bad.model.parameters(), 0.001)
            return result
        with patch.object(torch.Tensor, 'backward', wrong_backward):
            actual = bad.step(batch)
        log = io.StringIO()
        def equality_assertion():
            close(compared_state(reference), compared_state(bad), 'planted K4 state')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.FunctionTestCase(equality_assertion))
        self.assertFalse(result.wasSuccessful(), 'negative control must fail')
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 0)
        output = run_root()/'planted-per-micro-clipping'
        output.mkdir()
        (output/'test.log').write_text(log.getvalue())
        save_json(output/'RUN.json', {'schema': 'experiment', 'status': 'EXPECTED_FAIL',
            'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'unittest_success': result.wasSuccessful(), 'mutation': 'clip norm0.001 after each micro-batch backward',
            'source_sha256': code_identity(), 'log_sha256': sha(output/'test.log'), 'GPU_hours': 0,
            'interpretation': 'The equality assertion fails; the enclosing suite requires this failure and exits0.'})

    def test_fresh_process_resume_K1_K2_K4_exact_boundaries(self):
        for k in (1, 2, 4):
            roots = [run_root()/f'K{k}-{name}' for name in ('straight', 'split')]
            for phase, root in (('straight', roots[0]), ('first', roots[1]), ('second', roots[1]), ('resume', roots[1])):
                argv = [sys.executable, '-B', '-m', 'tests.c2_split_executor', phase, '--output', str(root), '--k', str(k)]
                child = subprocess.run(argv, capture_output=True, text=True, env=os.environ, timeout=400)
                (run_root()/f'K{k}-{phase}.log').write_text(child.stdout+child.stderr)
                self.assertEqual(child.returncode, 0, child.stderr)
            states = []
            for root in roots:
                ref = read_json(root/'LATEST.json')
                self.assertEqual(sha(root/ref['path']), ref['sha256'])
                states.append(torch.load(root/ref['path'], map_location='cpu', weights_only=False))
            for key in ('trainer', 'gradients', 'sampler', 'evaluations', 'rng', 'logs', 'model_training', 'identities'):
                exact(states[0][key], states[1][key], key)
            for name in ('metrics', 'samples', 'evaluations'):
                self.assertEqual((roots[0]/(name+'.jsonl')).read_bytes(), (roots[1]/(name+'.jsonl')).read_bytes())
            self.assertEqual(states[1]['sampler']['epoch'], 3)
            print(f'EXACT C2 resumed K={k}, 64 updates, splits u16/u32, all state and journals')

    def test_batch_identity_shape_and_K_guards(self):
        _, trainers = self.trainers(total=16)
        trainer = trainers[-1]
        batch = tensors(rows()[0])
        trainer.step(batch)
        altered = {k: v.flip(0) for k, v in batch.items()}
        with self.assertRaisesRegex(ValueError, 'identical'):
            trainer.step(altered)
        # experiment authorizes ACT8; retain the invalid-depth guard with ACT4.
        for eff, physical, steps in ((8, 3, 16), (8, 1, 16), (8, 2, 4), (0, 2, 16)):
            with self.assertRaises(ValueError):
                TrainingConfig(total_updates=16, effective_batch=eff, physical_batch=physical,
                               act_steps=steps, engineering_fixture=True)
