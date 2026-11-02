"""experiment boundary-only checkpoints and fail-closed RNG restoration."""
import io
import unittest
from unittest.mock import patch

import torch
from qdriver.common import INTERVAL, code_identity, save_json, sha
from qdriver.engine import Engine
from .c2_helpers import matched_tiny
from .helpers import exact, rows, run_root, tiny_config


class CheckpointTests(unittest.TestCase):
    def engine(self, name, config=None):
        return Engine(config or matched_tiny(2, 64), *rows(), run_root()/name, 'e'*64)

    def expected_failure(self, name, action, message):
        log = io.StringIO()
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.FunctionTestCase(action))
        self.assertFalse(result.wasSuccessful(), 'planted violation must fail')
        self.assertEqual(len(result.errors), 1)
        self.assertIn('ValueError: '+message, log.getvalue())
        output = run_root()/name
        output.mkdir()
        (output/'test.log').write_text(log.getvalue())
        save_json(output/'RUN.json', {'status': 'EXPECTED_FAIL', 'actual_exit': 1,
            'tests_run': result.testsRun, 'errors': 1, 'unittest_success': False,
            'source_sha256': code_identity(), 'log_sha256': sha(output/'test.log'), 'GPU_hours': 0,
            'scope': 'nested unittest action raises; enclosing suite requires failure and exits 0'})

    def test_boundaries_release_all_held_state_and_omit_payload(self):
        for config in (tiny_config(8, 32), tiny_config(16, 32), matched_tiny(1, 32),
                       matched_tiny(2, 32), matched_tiny(4, 32)):
            name = 'release-'+str(config['training']['act_steps'])+'-'+str(config['training'].get('effective_batch'))+'-'+str(config['training']['physical_batch'])
            engine = self.engine(name, config)
            engine.checkpoint()  # initialization is also a boundary
            steps = engine.trainer.config.act_steps
            self.assertEqual(INTERVAL % steps, 0)
            for _ in range(steps-1):
                engine.step()
                with self.assertRaisesRegex(ValueError, 'held-batch boundary'):
                    engine.checkpoint()
            engine.step()
            self.assertIsNone(engine.trainer.carry)
            self.assertIsNone(engine.trainer.masks)
            self.assertIsNone(engine.sampler.current)
            self.assertIsNone(engine.sampler.indices)
            state = engine.read_checkpoint(engine.checkpoint())
            self.assertFalse({'carry', 'masks', 'current_batch_identity'} & state['trainer'].keys())
            self.assertNotIn('indices', state['sampler'])
            before = engine.trainer.state_dict()
            engine.resume()
            exact(before, engine.trainer.state_dict())

    def test_planted_inside_batch_checkpoint_fails(self):
        engine = self.engine('inside-batch')
        ref = engine.checkpoint()
        engine.step()
        before = list((engine.output/'checkpoints').iterdir())
        self.expected_failure('planted-inside-batch', engine.checkpoint,
                              'checkpoint requires a held-batch boundary')
        self.assertEqual(before, list((engine.output/'checkpoints').iterdir()))
        self.assertEqual(engine.read_checkpoint(ref)['clocks']['updates'], 0)

    def test_live_state_at_boundary_refuses(self):
        engine = self.engine('live-state')
        for field in ('carry', 'masks', 'current_batch_identity'):
            setattr(engine.trainer, field, torch.ones(1))
            with self.assertRaisesRegex(ValueError, 'live held-batch state'):
                engine.checkpoint()
            setattr(engine.trainer, field, None)
        engine.trainer.model.step_noise = {'low': torch.ones(1)}
        with self.assertRaisesRegex(ValueError, 'live held-batch state'):
            engine.checkpoint()

    def test_next_batch_masks_and_noise_exact_after_resume(self):
        import qdriver.training as training
        engine = self.engine('next-draws')
        for _ in range(16):
            engine.step()
        ref = engine.checkpoint()
        draw = training._draw

        def next_state():
            noise = []
            def capture(*args, **kwargs):
                value = draw(*args, **kwargs)
                noise.append(value.clone())
                return value
            with patch.object(training, '_draw', side_effect=capture):
                metric = engine.step()
            self.assertEqual(len(noise), 3)
            return {'noise': noise, 'masks': engine.trainer.masks,
                    'batch': engine.sampler.current, 'sampler': engine.sampler.state_dict(),
                    'trainer': engine.trainer.state_dict(), 'metric': metric}

        expected = next_state()
        engine.resume(ref)
        actual = next_state()
        exact(expected, actual)
        save_json(run_root()/'NEXT_DRAWS.json', {'status': 'EXACT', 'update': 17,
            'comparisons': ['next batch', 'dropout masks', '3 keyed full-shape noise draws',
                            'metric', 'trainer', 'sampler'], 'GPU_hours': 0})

    def test_dropped_RNG_refuses_before_changing_live_state_or_journals(self):
        for key in ('cpu', 'cuda', 'python'):
            engine = self.engine('dropped-'+key)
            for _ in range(16):
                engine.step()
            ref = engine.checkpoint()
            state = engine.read_checkpoint(ref)
            del state['rng'][key]
            path = engine.output/ref['path']
            torch.save(state, path)
            bad = dict(ref, sha256=sha(path))  # bypass digest failure to exercise RNG validation
            engine.step()
            before = engine.trainer.state_dict()
            journals = {name: (engine.output/(name+'.jsonl')).read_bytes() for name in engine.journal.NAMES}
            self.expected_failure('planted-missing-RNG-'+key, lambda: engine.resume(bad),
                                  'checkpoint RNG state incomplete')
            exact(before, engine.trainer.state_dict())
            for name, body in journals.items():
                self.assertEqual((engine.output/(name+'.jsonl')).read_bytes(), body)
            self.assertFalse((engine.output/'RESUME.json').exists())
