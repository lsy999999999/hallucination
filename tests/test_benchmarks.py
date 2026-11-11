"""Check benchmark isolation and scoring on meaningful edge cases."""
import importlib.util
import json
from pathlib import Path
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmarks', ROOT / 'scripts/run_benchmarks.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def test_rgb_alias_groups_require_every_component(self):
        truth = [['Tampa', 'Tampa Bay'], ['Florida', 'FL']]
        self.assertTrue(bench.match_answer('TAMPA, Florida', truth))
        self.assertFalse(bench.match_answer('Tampa, Arizona', truth))
        self.assertFalse(bench.match_answer('', truth))

    def test_detection_all_positive_baseline_has_false_alarms(self):
        m = bench.detection_metrics([(True, True), (False, True), (False, True)])
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (1, 2, 0, 0))
        self.assertAlmostEqual(m['f1'], .5)

    def test_blind_inputs_and_distinct_ragtruth_sources(self):
        tasks = json.loads((bench.OUT / 'tasks.json').read_text())
        rag = json.loads((bench.OUT / 'ragtruth_samples_private.json').read_text())
        self.assertEqual(len(rag), len({r['source_id'] for r in rag}))
        self.assertEqual(len(tasks), len({t['id'] for t in tasks}))
        for task in tasks:
            expected = {'question', 'passages', 'candidate_answer'} if task['phase'] == 'detect' else {'question', 'documents'}
            self.assertEqual(set(task['input']), expected)

    def test_no_scoring_invalid_generated_output(self):
        for result in [dict(status='answer', answer='', factual_error_detected=False, reason=''),
                       dict(status='abstain', answer='Tampa', factual_error_detected=False, reason='')]:
            with self.assertRaises(AssertionError):
                bench.validate({'phase': 'generate'}, result)

    def test_fixed_pairing_and_noise_counts(self):
        rows = json.loads((bench.OUT / 'rgb_samples_private.json').read_text())
        base = [x for x in rows if x['subset'] == 'en_refine']
        for sid in {x['sample_id'] for x in base}:
            group = [x for x in base if x['sample_id'] == sid]
            self.assertEqual({x['condition'] for x in group}, {'direct', 'clean', 'mixed', 'irrelevant'})
            self.assertEqual(len({x['question'] for x in group}), 1)
            docs = {x['condition']: x['documents'] for x in group}
            self.assertEqual(len(docs['direct']), 0)
            self.assertEqual(sum((Counter(docs['mixed']) & Counter(docs['clean'])).values()), 2)
            self.assertEqual(sum((Counter(docs['mixed']) & Counter(docs['irrelevant'])).values()), 3)


if __name__ == '__main__':
    unittest.main()
