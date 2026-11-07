import json
import tempfile
import unittest
from pathlib import Path

from hallulab.gold import validate_gold, retrieval_diagnostic
from hallulab.evaluation import export_review
from hallulab.io_utils import write_jsonl, load_csv


class GoldIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = Path(self.temp.name)
        self.corpus = self.p/'corpus.jsonl'
        self.questions = self.p/'questions.jsonl'
        self.gold = self.p/'gold.jsonl'
        self.experiment = self.p/'experiment.jsonl'
        write_jsonl(self.corpus, [
            {'id': 'a_p001_c00', 'paper_id': 'a', 'page': 1, 'title': 'Alpha', 'text': 'The date of the event is 2024.'},
            {'id': 'a_p002_c00', 'paper_id': 'a', 'page': 2, 'title': 'Alpha', 'text': 'Another unrelated text.'},
        ])
        write_jsonl(self.questions, [
            {'id': 'Q1', 'type': 'single_fact', 'question': 'What is the date of the event?', 'target_papers':['a']},
            {'id': 'Q2', 'type': 'unanswerable', 'question': 'What is the hidden number?', 'target_papers':['a']},
        ])
        write_jsonl(self.gold, [
            {'id': 'Q1', 'type': 'single_fact', 'question': 'What is the date of the event?', 'target_papers':['a'],
             'answerable':True, 'gold_answer':'2024', 'gold_claims':['The date is 2024.'],
             'gold_evidence_ids':['a_p001_c00'], 'annotation_status':'source_checked_draft_needs_independent_review'},
            {'id': 'Q2', 'type': 'unanswerable', 'question': 'What is the hidden number?', 'target_papers':['a'],
             'answerable':False, 'gold_answer':'Cannot determine.', 'gold_claims':[],
             'gold_evidence_ids':[], 'annotation_status':'source_checked_draft_needs_independent_review'}
        ])
        write_jsonl(self.experiment,[
            {'question_id':'Q1','question':'What is the date of the event?', 'question_type':'single_fact',
             'retrieved':[{'id':'a_p001_c00'}],
             'A':{'answer':'2024', 'abstained':False,'claims':[{'text':'2024','citations':[]}]},
             'B':{'answer':'2024', 'abstained':False,'claims':[{'text':'2024','citations':['a_p001_c00']}]},
             'C':{'answer':'2024', 'abstained':False,'claims':[{'text':'2024','citations':['a_p001_c00']}]}}
        ])

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_provisional_gold(self):
        report=validate_gold(self.gold,self.questions,self.corpus)
        self.assertTrue(report['valid'])
        self.assertEqual(report['answerable'],1)
        self.assertEqual(report['independently_verified'],0)

    def test_missing_evidence_fails(self):
        rows=[json.loads(l) for l in self.gold.read_text().splitlines()]
        rows[0]['gold_evidence_ids']=['a_p099_c00']
        write_jsonl(self.gold,rows)
        report=validate_gold(self.gold,self.questions,self.corpus)
        self.assertFalse(report['valid'])
        self.assertIn('missing evidence ID',str(report['errors']))

    def test_mismatch_question_fails(self):
        rows=[json.loads(l) for l in self.gold.read_text().splitlines()]
        rows[0]['question']='changed!'
        write_jsonl(self.gold,rows)
        self.assertFalse(validate_gold(self.gold,self.questions,self.corpus)['valid'])

    def test_retrieval_and_unanswerable_exclusion(self):
        summary=retrieval_diagnostic(self.gold,self.questions,self.corpus,self.p/'diagnostic',top_k=1)
        self.assertEqual(summary['answerable_questions'],1)
        self.assertEqual(summary['unanswerable_questions_excluded'],1)
        self.assertAlmostEqual(summary['micro_evidence_recall_at_k'],1.0)
        self.assertTrue((self.p/'diagnostic/retrieval_per_question.csv').exists())

    def test_review_prefill_no_false_human_truth_and_no_overwrite(self):
        export_review(self.experiment,self.p/'review',gold_path=self.gold)
        q=load_csv(self.p/'review/questions_review.csv')
        self.assertEqual(q[0]['gold_evidence_ids'],'a_p001_c00')
        self.assertEqual(q[0]['gold_claims_total'],'1')
        self.assertEqual(q[0]['draft_answerable_in_corpus'],'1')
        self.assertEqual(q[0]['human_answerable_in_corpus'],'')
        self.assertEqual(q[0]['gold_claims_covered'],'')
        with self.assertRaises(FileExistsError):
            export_review(self.experiment,self.p/'review',gold_path=self.gold)

if __name__=='__main__':
    unittest.main()
