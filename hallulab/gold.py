"""Gold annotation validation and independent retrieval diagnostics.

Gold is ONLY used in post-run evaluation and retrieval diagnostics; never in
model generation or retrieval-query construction. Draft annotations are not
human adjudicated truth.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_csv

REVIEWED_STATUSES = {'independently_verified', 'human_verified', 'final_verified'}


def validate_gold(gold_path, questions_path, corpus_path):
    gold = read_jsonl(gold_path)
    questions = read_jsonl(questions_path)
    corpus = read_jsonl(corpus_path)
    errors, warnings = [], []
    q_ids = [q.get('id') for q in questions]
    g_ids = [g.get('id') for g in gold]
    c_ids = [c.get('id') for c in corpus]

    for label, ids in [('questions', q_ids), ('gold', g_ids), ('corpus', c_ids)]:
        duplicates = [key for key, n in Counter(ids).items() if n > 1]
        if duplicates:
            errors.append(f'Duplicate IDs in {label}: {duplicates[:10]}')

    question_map = {q['id']: q for q in questions}
    corpus_map = {c['id']: c for c in corpus}
    missing_gold = sorted(set(q_ids) - set(g_ids))
    extra_gold = sorted(set(g_ids) - set(q_ids))
    if missing_gold: errors.append(f'Questions missing Gold: {missing_gold}')
    if extra_gold: errors.append(f'Gold IDs not in questions: {extra_gold}')

    for g in gold:
        qid = g.get('id', '<unknown>')
        q = question_map.get(qid)
        if not q:
            continue
        if g.get('question') != q.get('question'):
            errors.append(f'{qid}: question text differs from experiment input')
        if g.get('type') != q.get('type'):
            errors.append(f'{qid}: question type differs')
        if g.get('target_papers') != q.get('target_papers'):
            warnings.append(f'{qid}: target paper list differs')

        answerable = g.get('answerable')
        claims = g.get('gold_claims', [])
        evidence_ids = g.get('gold_evidence_ids', [])
        if not isinstance(answerable, bool):
            errors.append(f'{qid}: answerable is not a boolean')
        elif answerable and (not claims or not evidence_ids or not g.get('gold_answer', '').strip()):
            errors.append(f'{qid}: answerable but missing gold claims, answer or evidence')
        elif answerable is False and (claims or evidence_ids):
            errors.append(f'{qid}: unanswerable but has claims/evidence')
        if not isinstance(claims, list) or not all(isinstance(c, str) and c.strip() for c in claims):
            errors.append(f'{qid}: invalid gold_claims')
        if not isinstance(evidence_ids, list):
            errors.append(f'{qid}: invalid gold_evidence_ids')
            continue
        for cid in evidence_ids:
            c = corpus_map.get(cid)
            if not c:
                errors.append(f'{qid}: missing evidence ID {cid}')
            elif c.get('paper_id') not in g.get('target_papers', []):
                warnings.append(f'{qid}: evidence {cid} comes from a non-target paper')
        status = g.get('annotation_status', '')
        if status not in REVIEWED_STATUSES:
            warnings.append(f'{qid}: Gold still provisional ({status or "no status"})')

    n_answerable = sum(g.get('answerable') is True for g in gold)
    n_unanswerable = sum(g.get('answerable') is False for g in gold)
    reviewed = sum(g.get('annotation_status') in REVIEWED_STATUSES for g in gold)
    return {
        'valid': not errors,
        'gold_questions': len(gold),
        'experiment_questions': len(questions),
        'corpus_chunks': len(corpus),
        'corpus_papers': len({c.get('paper_id') for c in corpus}),
        'answerable': n_answerable,
        'unanswerable': n_unanswerable,
        'independently_verified': reviewed,
        'provisional': len(gold) - reviewed,
        'errors': errors,
        'warnings': warnings
    }


def retrieval_diagnostic(gold_path, questions_path, corpus_path, out_dir,
                         top_k=4, retriever='bm25'):
    if top_k <= 0:
        raise ValueError('--k must be positive')
    report = validate_gold(gold_path, questions_path, corpus_path)
    if not report['valid']:
        raise ValueError('Invalid Gold annotations: ' + '; '.join(report['errors']))
    gold = read_jsonl(gold_path)
    corpus = read_jsonl(corpus_path)
    if retriever == 'bm25':
        from .retrieval import BM25Retriever
        finder = BM25Retriever(corpus)
    elif retriever == 'hybrid':
        from .retrieval import HybridRetriever
        finder = HybridRetriever(corpus)
    else:
        raise ValueError('Unknown retriever')
    rows = []
    total_gold, total_hits, any_hits, complete_hits = 0, 0, 0, 0
    by_type = defaultdict(lambda: {'questions': 0, 'gold_ids': 0, 'found_ids': 0})
    for g in gold:
        # This uses only the literal question as retrieval input, not the gold answer.
        found = finder.search(g['question'], top_k=top_k)
        got = [r['id'] for r in found]
        expected = g['gold_evidence_ids']
        expected_set = set(expected)
        found_ids = expected_set.intersection(got)
        ratio = len(found_ids) / len(expected_set) if expected_set else None
        if g['answerable']:
            total_gold += len(expected_set)
            total_hits += len(found_ids)
            any_hits += bool(found_ids)
            complete_hits += expected_set.issubset(got)
            by_type[g['type']]['questions'] += 1
            by_type[g['type']]['gold_ids'] += len(expected_set)
            by_type[g['type']]['found_ids'] += len(found_ids)
        rows.append({
            'question_id': g['id'], 'type': g['type'],
            'answerable_gold_draft': int(g['answerable']),
            'gold_annotation_status': g['annotation_status'],
            'gold_evidence_count': len(expected_set),
            'found_count': len(found_ids),
            'evidence_recall_at_k': round(ratio, 4) if ratio is not None else '',
            'any_hit': int(bool(found_ids)) if expected_set else '',
            'all_hit': int(expected_set.issubset(got)) if expected_set else '',
            'retrieved_ids': ' '.join(got),
            'gold_ids': ' '.join(expected),
            'missed_gold_ids': ' '.join(cid for cid in expected if cid not in got),
        })

    answerable_rows = [r for r in rows if r['answerable_gold_draft']]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'retrieval_per_question.csv', rows, list(rows[0]) if rows else ['question_id'])
    summary = {
        'experimental_retriever': retriever,
        'k': top_k,
        'answerable_questions': len(answerable_rows),
        'unanswerable_questions_excluded': len(rows) - len(answerable_rows),
        'micro_evidence_recall_at_k': total_hits / total_gold if total_gold else None,
        'macro_evidence_recall_at_k': sum(r['evidence_recall_at_k'] for r in answerable_rows) / len(answerable_rows) if answerable_rows else None,
        'any_evidence_hit_rate': any_hits / len(answerable_rows) if answerable_rows else None,
        'all_gold_evidence_hit_rate': complete_hits / len(answerable_rows) if answerable_rows else None,
        'by_type': {key: {**v, 'micro_recall': v['found_ids'] / v['gold_ids'] if v['gold_ids'] else None} for key,v in sorted(by_type.items())},
        'gold_annotation_status': 'PROVISIONAL' if report['provisional'] else 'INDEPENDENTLY_VERIFIED',
        'notes': [
            'Gold IDs are provisional; retrieval values are diagnostic estimates until independently checked.',
            'Exact-ID recall penalizes alternative valid passages and multi-hop evidence with incomplete gold labels.',
            'Unanswerable questions are excluded from evidence recall denominator.',
            'Retriever received only question text, never the gold answer or gold evidence.'
        ]
    }
    (out/'retrieval_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
