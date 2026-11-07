"""File handoff for actual native Codex inference; no simulated model responses."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallulab.experiment import GEN_INSTRUCTIONS, VERIFY_INSTRUCTIONS, clean_generated, filter_supported
from hallulab.io_utils import read_jsonl, write_jsonl
from hallulab.retrieval import BM25Retriever
from hallulab.evaluation import export_review

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/native_codex_20261107_001'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    questions = read_jsonl(ROOT / 'data/questions.jsonl')
    corpus = read_jsonl(ROOT / 'data/corpus.jsonl')
    retriever = BM25Retriever(corpus)
    sources = {q['id']: retriever.search(q['question'], top_k=4) for q in questions}
    save(RUN / 'retrieval.json', sources)
    save(RUN / 'protocol.json', {
        'backend': 'native_codex_sessions', 'model': 'inherited native Codex model; exact serving identifier unverified',
        'batch_size': 5, 'questions': len(questions), 'retriever': 'bm25', 'k': 4,
        'corpus_sha256': digest(ROOT / 'data/corpus.jsonl'),
        'questions_sha256': digest(ROOT / 'data/questions.jsonl'),
        'generation_instructions_sha256': hashlib.sha256(GEN_INSTRUCTIONS.encode()).hexdigest(),
        'verification_instructions_sha256': hashlib.sha256(VERIFY_INSTRUCTIONS.encode()).hexdigest(),
        'exposure': 'Fresh no-history sessions per batch and condition; five questions share one session. Agents may only read their assigned input packet and write their output. No API or website calls.',
        'C': 'deterministic filtering of the same B output, using separate verifier sessions',
        'evaluation': 'Automatic verifier labels are not human factuality gold. Human review remains unfilled.',
        'temperature_control': False,
    })
    for group in ['A', 'B']:
        for batch, start in enumerate(range(0, len(questions), 5), 1):
            items = []
            for q in questions[start:start + 5]:
                if group == 'A':
                    request = f"Question: {q['question']}\n\nNo sources available. Give your own best factual answer or abstain if uncertain."
                else:
                    context = '\n\n'.join(f"[ID: {x['id']} | {x['title']} | PDF page {x['page']}]\n{x['text']}" for x in sources[q['id']])
                    request = f"Question: {q['question']}\n\nSources (ONLY these may be cited):\n{context}\n\nProduce an evidence-grounded answer. If sources cannot support the answer, abstain."
                items.append({'question_id': q['id'], 'request': request})
            save(RUN / f'inputs/{group}{batch:02}.json', {'instructions': GEN_INSTRUCTIONS, 'items': items,
                'output_format': {'responses': [{'question_id': 'Qxx', 'answer': '...', 'abstained': False,
                    'claims': [{'text': 'one atomic fact', 'citations': []}]}]}})
    print('Prepared 12 generation packets, five questions per independent session.')


def generated(group):
    result = {}
    for path in sorted((RUN / 'inputs').glob(f'{group}*.json')):
        expected = [q['question_id'] for q in json.loads(path.read_text())['items']]
        out = json.loads((RUN / 'outputs' / path.name).read_text())
        responses = out['responses']
        assert [r['question_id'] for r in responses] == expected, (path.name, 'question coverage/order')
        for row in responses:
            assert isinstance(row['abstained'], bool)
            assert isinstance(row['answer'], str) and isinstance(row['claims'], list)
            assert (row['abstained'] and not row['claims']) or (not row['abstained'] and 1 <= len(row['claims']) <= 5)
            for c in row['claims']:
                assert isinstance(c['text'], str) and c['text'].strip()
                assert isinstance(c['citations'], list) and all(isinstance(v, str) for v in c['citations'])
                if group == 'A':
                    assert c['citations'] == [], 'Direct generation must not cite sources'
            result[row['question_id']] = clean_generated(row)
    assert len(result) == 30
    return result


def prepare_verification():
    generated('A')
    b = generated('B')
    sources = json.loads((RUN / 'retrieval.json').read_text())
    automatic = {}
    for batch, start in enumerate(range(0, 30, 5), 1):
        jobs = []
        for qid in list(b)[start:start + 5]:
            byid = {s['id']: s for s in sources[qid]}
            for i, c in enumerate(b[qid]['claims']):
                cid = f'{qid}:{i}'
                cited = list(dict.fromkeys(c['citations']))
                if not cited or any(x not in byid for x in cited):
                    automatic[cid] = {'claim_id': cid, 'label': 'INSUFFICIENT', 'reason': 'No citations or fabricated/unretrieved source ID'}
                else:
                    evidence = '\n\n'.join(f"[{x}] {byid[x]['text']}" for x in cited)
                    jobs.append({'claim_id': cid, 'claim': c['text'], 'cited_evidence': evidence})
        save(RUN / f'inputs/V{batch:02}.json', {'instructions': VERIFY_INSTRUCTIONS, 'items': jobs})
        print(f'V{batch:02}: {len(jobs)} claim entailment jobs')
        if not jobs:
            save(RUN / f'outputs/V{batch:02}.json', {'verdicts': []})
    save(RUN / 'automatic_verdicts.json', automatic)


def finish():
    a, b = generated('A'), generated('B')
    sources = json.loads((RUN / 'retrieval.json').read_text())
    verdicts = json.loads((RUN / 'automatic_verdicts.json').read_text())
    for path in sorted((RUN / 'inputs').glob('V*.json')):
        expected = {x['claim_id'] for x in json.loads(path.read_text())['items']}
        rows = json.loads((RUN / 'outputs' / path.name).read_text())['verdicts']
        assert len(rows) == len(expected) and {v['claim_id'] for v in rows} == expected
        for v in rows:
            assert v['label'] in {'SUPPORTED', 'CONTRADICTED', 'INSUFFICIENT'}
            assert isinstance(v['reason'], str) and v['reason'].strip()
            verdicts[v['claim_id']] = v
    rows = []
    for q in read_jsonl(ROOT / 'data/questions.jsonl'):
        qid = q['id']
        vs = [dict(verdicts[f'{qid}:{i}'], claim_id=str(i)) for i in range(len(b[qid]['claims']))]
        c = filter_supported(b[qid], vs)
        rows.append({'question_id': qid, 'question_type': q['type'], 'question': q['question'],
            'retrieved': [{k: x[k] for k in ['id', 'paper_id', 'page', 'score']} for x in sources[qid]],
            'A': a[qid], 'B': b[qid], 'C': c, 'verdicts_B': vs,
            'b_verifier_support_rate': sum(v['label'] == 'SUPPORTED' for v in vs) / len(vs) if vs else None,
            'backend': 'native_codex_sessions',
            'native_packets': {g: f'{g}{(int(qid[1:])-1)//5+1:02}.json' for g in ['A', 'B', 'V']}})
    write_jsonl(RUN / 'experiment.jsonl', rows)
    summary = {'questions_completed': len(rows), 'groups': {},
        'B_verdict_counts': dict(Counter(v['label'] for r in rows for v in r['verdicts_B'])),
        'status': 'generation_and_filtering_complete; independent_human_evaluation_pending'}
    for g in ['A', 'B', 'C']:
        summary['groups'][g] = {'claims': sum(len(r[g]['claims']) for r in rows),
            'answered_questions': sum(not r[g]['abstained'] for r in rows),
            'abstained_questions': sum(r[g]['abstained'] for r in rows)}
    summary['claims_removed_from_B'] = summary['groups']['B']['claims'] - summary['groups']['C']['claims']
    export_review(RUN / 'experiment.jsonl', RUN / 'review')
    save(RUN / 'SUMMARY.json', summary)
    files = sorted(p for d in ['inputs', 'outputs'] for p in (RUN / d).glob('*.json'))
    save(RUN / 'artifact_hashes.json', {str(p.relative_to(RUN)): digest(p) for p in files})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'verify', 'finish'])
    {'prepare': prepare, 'verify': prepare_verification, 'finish': finish}[parser.parse_args().stage]()
