"""Fixed experiment scientific stops and arithmetic-only calibration selection."""
import math
from .common import CAP_SECONDS, INTERVAL, PLAN_SECONDS, require


def flat_at(evaluation_number, gain_pp):
    """A1.2: unrounded percentage-point gain over ten evaluation intervals."""
    return evaluation_number >= 11 and gain_pp < 1.0


class StopRules:
    def __init__(self, evaluations=(), *, stop_on_flat=True):
        self.evaluations = list(evaluations)
        self.stop_on_flat = stop_on_flat

    @property
    def best(self):
        return max((x['strict_correct'] for x in self.evaluations), default=None)

    def observe(self, update, correct, total=2048):
        require(type(update) is int and update > 0 and update % INTERVAL == 0, 'evaluation cadence')
        require(total == 2048 and type(correct) is int and 0 <= correct <= total, 'strict ID count')
        require(not self.evaluations or update == self.evaluations[-1]['update']+INTERVAL, 'evaluation gap/order')
        self.evaluations.append({'update': update, 'strict_correct': correct, 'total': total})
        return self.scientific()

    def scientific(self):
        if not self.evaluations:
            return None
        last = self.evaluations[-1]
        if last['strict_correct'] >= 1844:
            return {'status': 'REACHED', 'updates_to_90_percent': last['update'], 'strict_correct': last['strict_correct'], 'total': 2048}
        return self.flat_record() if self.stop_on_flat else None

    def flat_record(self):
        if not self.evaluations:
            return None
        last = self.evaluations[-1]
        start = last['update'] - 20480
        earlier = [x['strict_correct'] for x in self.evaluations if x['update'] <= start]
        # Ten 2048-update intervals. With no evaluation at update zero the first
        # defined comparison is update 22528 against update 2048.
        if earlier and flat_at(len(self.evaluations), (self.best-max(earlier))*100/2048):
            return {'status': 'FLAT', 'plateau_strict_correct': self.best, 'total': 2048,
                    'window_start_update': start, 'window_end_update': last['update'],
                    'running_max_gain_correct': self.best-max(earlier)}
        return None

    def schedule_end(self, completed_updates):
        last = self.evaluations[-1] if self.evaluations else None
        baseline = self.evaluations[:-10]
        gain = self.best-max(x['strict_correct'] for x in baseline) if baseline else None
        return {'status': 'SCHEDULE_END', 'reached': False, 'completed_updates': completed_updates,
                'best_strict_correct': self.best, 'final_strict_correct': None if last is None else last['strict_correct'],
                'total': 2048, 'evaluation_count': len(self.evaluations),
                'running_max_gain_last_ten_correct': gain,
                'running_max_gain_last_ten_pp': None if gain is None else gain*100/2048,
                'final_gain_last_ten_correct': None if not baseline else last['strict_correct']-baseline[-1]['strict_correct'],
                'ten_interval_change_available': bool(baseline),
                'reason': 'not reached within T; no continuation without a new PI decision'}

    def cap(self, charged_seconds):
        require(math.isfinite(charged_seconds) and charged_seconds >= 0, 'bad charge')
        if charged_seconds >= CAP_SECONDS:
            return {'status': 'CAP', 'reached': False, 'best_strict_correct': self.best, 'total': 2048}
        return None


def timing_inputs(candidate):
    require(candidate.get('status') == 'PASS', 'selected calibration did not complete')
    require(type(candidate.get('peak_reserved_bytes')) is int and candidate['peak_reserved_bytes'] >= 0,
            'peak memory must be measured')
    values = {k: candidate[k] for k in ('seconds_per_update', 'evaluation_seconds', 'checkpoint_seconds', 'fixed_overhead_seconds')}
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())
            and values['seconds_per_update'] > 0, 'nonfinite calibration timings')
    require(type(candidate.get('checkpoint_bytes')) is int and candidate['checkpoint_bytes'] > 0
            and type(candidate.get('evaluation_record_bytes')) is int and candidate['evaluation_record_bytes'] > 0,
            'incomplete calibration storage measurement')
    require(all(type(candidate.get(k)) in (int, float) and math.isfinite(candidate[k]) and candidate[k] >= 0
                for k in ('logs_bytes_per_update', 'metrics_bytes_per_update')), 'incomplete calibration log measurement')
    return values


def choose_schedule(calibration, charged_seconds=0, *, fixed_T=None, cap_seconds=CAP_SECONDS,
                    reserve_seconds=CAP_SECONDS-PLAN_SECONDS):
    if calibration['schema'] == 'experiment':
        return choose_matched_schedule(calibration, charged_seconds, fixed_T=fixed_T,
                                       cap_seconds=cap_seconds, reserve_seconds=reserve_seconds)
    require(calibration['schema'] == 'experiment', 'calibration schema')
    candidates = calibration['candidates']
    model_mode = calibration.get('model_mode', 'recursive')
    require(model_mode in ('recursive', 'nonrecursive'), 'unknown calibration model')
    if fixed_T is not None or model_mode == 'nonrecursive':
        require(fixed_T is None or (type(fixed_T) is int and fixed_T == 196608), 'standard fixed T must equal 196608; other T requires recipe-matched mode')
        require([(x['batch'], x['steps']) for x in candidates] == [(1024, 8)], 'standard fixed T calibrates only batch1024/ACT8')
        require(candidate_fits(candidates[0]), 'standard calibration must complete within 44 GB')
        result = project_schedule(candidates[0], charged_seconds, fixed_T=fixed_T,
                                  cap_seconds=cap_seconds, reserve_seconds=reserve_seconds)
        result.update(selection_reason='R65-original_configuration_and_T_fixed_memory_throughput_storage_checks_only', preferred_failure=None)
        if model_mode == 'nonrecursive':
            result['model_mode'] = model_mode
        return result
    require([(x['batch'], x['steps']) for x in candidates] == [(4096, 16), (1024, 8)], 'two fixed calibration candidates')
    preferred, fallback = candidates
    require(type(charged_seconds) in (int, float) and math.isfinite(charged_seconds) and charged_seconds >= 0, 'bad calibration charge')
    peak = preferred.get('peak_reserved_bytes')
    failure = None
    try:
        require(type(peak) is int and peak >= 0, 'preferred peak memory must be measured')
        timing_inputs(preferred)
    except (ValueError, KeyError, TypeError) as error:
        failure = str(error)
    # GB is decimal as written in DECISION_experiment; this is disclosed in README.
    selected = preferred if failure is None and peak <= 44_000_000_000 else fallback
    result = project_schedule(selected, charged_seconds, cap_seconds=cap_seconds, reserve_seconds=reserve_seconds)
    result.update(selection_reason='preferred_calibration_failed' if failure is not None else 'preferred_memory_fits' if selected is preferred else 'preferred_memory_exceeds_threshold',
                  preferred_failure=failure)
    return result


def candidate_fits(candidate):
    try:
        timing_inputs(candidate)
        return candidate['peak_reserved_bytes'] <= 44_000_000_000
    except (ValueError, KeyError, TypeError):
        return False


def choose_matched_schedule(calibration, charged_seconds=0, **options):
    candidates = calibration['candidates']
    steps = calibration.get('act_steps', 16)
    require(type(steps) is int and steps in (8, 16), 'accumulation ACT steps must be 8 or 16')
    expected = [(4096, steps, 4096, 1), (2048, steps, 4096, 2), (1024, steps, 4096, 4)]
    if options.get('fixed_T') == 38912 and steps == 16:
        expected = [(2048, 16, 4096, 2)]  # experiment keeps R524-B4096-D16-short's measured micro-batch.
    actual = [(x['batch'], x['steps'], x['effective_batch'], x['accumulation_steps']) for x in candidates]
    require(0 < len(actual) <= 3 and actual == expected[:len(actual)], 'ordered C2 calibration candidates')
    selected = next((x for x in candidates if candidate_fits(x)), None)
    require(selected is not None, 'no C2 micro-batch completed within 44 GB; no reduced-depth fallback')
    result = project_schedule(selected, charged_seconds, **options)
    result.update(effective_batch=4096, accumulation_steps=selected['accumulation_steps'],
                  selection_reason=f'first_complete_ACT{steps}_microbatch_within_44GB')
    return result


def project_schedule(selected, charged_seconds, *, fixed_T=None, cap_seconds=CAP_SECONDS,
                     reserve_seconds=CAP_SECONDS-PLAN_SECONDS):
    require(type(charged_seconds) in (int, float) and math.isfinite(charged_seconds) and charged_seconds >= 0, 'bad calibration charge')
    require(type(cap_seconds) in (int, float) and math.isfinite(cap_seconds) and cap_seconds > 0, 'bad cap seconds')
    require(type(reserve_seconds) in (int, float) and math.isfinite(reserve_seconds)
            and 0 <= reserve_seconds < cap_seconds, 'bad reserve seconds')
    require(fixed_T is None or (type(fixed_T) is int and fixed_T >= INTERVAL and fixed_T % INTERVAL == 0),
            'fixed T must be a positive multiple of 2048')
    planning_seconds = cap_seconds-reserve_seconds
    available = planning_seconds-charged_seconds
    timings = timing_inputs(selected)
    s = selected['seconds_per_update']
    e, c = selected['evaluation_seconds'], selected['checkpoint_seconds']
    fixed = selected['fixed_overhead_seconds']
    per_block = s*INTERVAL + e+c
    # Correct any boundary rounding with the actual projected-time expression.
    def projection(n):
        return s*(n*INTERVAL) + n*(e+c) + fixed
    if fixed_T is not None:
        blocks = fixed_T//INTERVAL
        require(projection(blocks) <= available, 'fixed T projected charge exceeds cap minus reserve')
    else:
        blocks = math.floor((available-fixed)/per_block)
        require(blocks >= 1, 'no full 2048-update block fits planning budget')
        while blocks and projection(blocks) > available:
            blocks -= 1
        while projection(blocks+1) <= available:
            blocks += 1
        require(blocks >= 1, 'no full block fits')
    result = {'batch': selected['batch'], 'steps': selected['steps'], 'T': blocks*INTERVAL,
            'projected_seconds': projection(blocks), 'next_block_projected_seconds': projection(blocks+1),
            'planning_seconds': planning_seconds, 'cap_seconds': cap_seconds,
            'calibration_charged_seconds': charged_seconds, 'available_main_seconds': available,
            'projected_total_charged_seconds': charged_seconds+projection(blocks),
            'reserve_seconds': reserve_seconds, 'memory_threshold_bytes': 44_000_000_000,
            'timing_inputs': timings}
    if fixed_T is not None:
        result['fixed_T'] = fixed_T
    return result
