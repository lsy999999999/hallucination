import math
import tempfile
import unittest
from pathlib import Path
from hallulab.corpus import chunks_from_pages
from hallulab.retrieval import BM25Retriever
from hallulab.experiment import clean_generated, filter_supported, run_one, verify_claims
from hallulab.uncertainty import discrete_entropy
from hallulab.io_utils import write_jsonl, read_jsonl
from hallulab.evaluation import export_review,evaluate_review


class FakeLLM:
    def json(self,sys,user,temperature=None):
        if 'entailment verifier' in sys:
            return {'verdicts':[{'claim_id':'0','label':'SUPPORTED','reason':'evidence'}]}
        return {'answer':'Alpha was established in 2020.', 'abstained':False,
                'claims':[{'text':'Alpha was established in 2020.',
                           'citations':['alpha_p001_c00'] if 'Sources (ONLY' in user else []}]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.corpus=[{'id':'alpha_p001_c00','paper_id':'alpha','page':1,'title':'Alpha',
                      'text':'Alpha was established in 2020. Alpha is a fictional research institute.'},
                     {'id':'beta_p001_c00','paper_id':'beta','page':1,'title':'Beta',
                      'text':'Beta has no relationship with Alpha.'}]

    def test_retriever(self):
        got=BM25Retriever(self.corpus).search('when was Alpha established', top_k=1)
        self.assertEqual(got[0]['id'],'alpha_p001_c00')

    def test_triplet(self):
        r=run_one({'id':'T1','type':'single_fact','question':'When was Alpha established?'},
                  BM25Retriever(self.corpus),FakeLLM(),FakeLLM(),top_k=2)
        self.assertEqual(r['A']['claims'][0]['citations'],[])
        self.assertEqual(r['verdicts_B'][0]['label'],'SUPPORTED')
        self.assertEqual(len(r['C']['claims']),1)

    def test_bogus_citation_not_verified(self):
        generated={'claims':[{'text':'bogus','citations':['nonexistent']}], 'abstained':False}
        verdicts=verify_claims(FakeLLM(),generated,self.corpus)
        self.assertEqual(verdicts[0]['label'],'INSUFFICIENT')
        self.assertTrue(filter_supported(generated,verdicts)['abstained'])

    def test_chunking(self):
        chunks=chunks_from_pages('a',[' '.join(['hello']*500)],210,35)
        self.assertGreaterEqual(len(chunks),2)
        self.assertTrue(chunks[0]['id'].startswith('a_p001'))

    def test_entropy(self):
        self.assertAlmostEqual(discrete_entropy([['a']*3,['b']*2]),0.673011667,places=6)

    def test_export_review_and_human_eval(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'runs.jsonl';out=Path(t)/'review'
            r=run_one({'id':'T1','type':'single_fact','question':'When was Alpha established?'},
                      BM25Retriever(self.corpus),FakeLLM(),FakeLLM(),2)
            write_jsonl(source,[r]);export_review(source,out)
            from hallulab.io_utils import load_csv,write_csv
            claim_path=out/'claims_review.csv'
            rows=load_csv(claim_path)
            for row in rows:
                row['human_factual_correct']='1'
                if row['group']!='A': row['human_citation_supported']='1'
            write_csv(claim_path,rows,rows[0].keys())
            use_path=out/'citations_review.csv'; use=load_csv(use_path)
            for row in use: row['human_citation_useful']='1'
            write_csv(use_path,use,use[0].keys())
            qpath=out/'questions_review.csv'; questions=load_csv(qpath)
            for row in questions:
                row['human_answerable_in_corpus']='1'
                row['gold_claims_total']='1'
                row['gold_claims_covered']='1'
                row['gold_evidence_ids']='alpha_p001_c00'
            write_csv(qpath,questions,questions[0].keys())
            metrics=evaluate_review(claim_path,qpath,citation_csv=use_path)
            self.assertAlmostEqual(metrics['A']['factual_precision'],1.)
            self.assertIsNone(metrics['A']['citation_support_rate'])
            self.assertAlmostEqual(metrics['B']['citation_usefulness_precision'],1.)
            self.assertAlmostEqual(metrics['C']['evidence_recall_at_k'],1.)
            self.assertAlmostEqual(metrics['C']['claim_f1'],1.)

if __name__=='__main__': unittest.main()
