"""Schedule schema checks using only the standard library; no model execution."""
import json
from pathlib import Path
import unittest
from qdriver.workload import config_for_schedule, validate_config

BASE = Path(__file__).resolve().parents[2]


class ScheduleTests(unittest.TestCase):
    def test_every_shipped_schedule(self):
        schedules = sorted((BASE/'configs').glob('*_schedule.json'))
        self.assertEqual(len(schedules), 10)
        for path in schedules:
            with self.subTest(schedule=path.name):
                schedule = json.loads(path.read_text())
                config = config_for_schedule(schedule)
                validate_config(config)
                training = config['training']
                self.assertEqual(training['act_steps'], schedule['steps'])
                self.assertEqual(training['physical_batch'], schedule['batch'])
                self.assertEqual(training['total_updates'], schedule['T'])
                if 'effective_batch' in schedule:
                    self.assertEqual(training['effective_batch'], schedule['effective_batch'])

    def test_act_steps_alias(self):
        config = config_for_schedule(dict(batch=1024, act_steps=16, T=2048))
        self.assertEqual(config['training']['act_steps'], 16)

    def test_steps_takes_precedence(self):
        config = config_for_schedule(dict(batch=1024, steps=8, act_steps=16, T=2048))
        self.assertEqual(config['training']['act_steps'], 8)

    def test_missing_steps_rejected(self):
        with self.assertRaises(KeyError):
            config_for_schedule(dict(batch=1024, T=2048))


if __name__ == '__main__':
    unittest.main()
