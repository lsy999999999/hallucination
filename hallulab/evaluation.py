"""Human evaluation. No judge-generated labels masquerade as gold truth."""
import json
from collections import defaultdict
from pathlib import Path

from .io_utils import load_csv, read_jsonl, write_csv


def export_review(run_path, out_dir, gold_path=None, overwrite=False):
    rows = read_jsonl(run_path)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs=[directory/n for n in ('claims_review.csv','questions_review.csv','citations_review.csv')]
    if not overwrite and any(f.exists() for f in outputs):
        raise FileExistsError('Existing human review files found. Use another --out-dir, or --overwrite only if you intend to discard prior labels.')
    gold = {r['id']:r for r in read_jsonl(gold_path)} if gold_path else {}
    claim_rows, q_rows, citation_rows = [], [], []
    for r in rows:
        for group in ['A','B','C']:
            obj = r[group]
            g=gold.get(r['question_id'],{})
            q_rows.append({'question_id':r['question_id'], 'group':group,
                'question':r['question'],'question_type':r['question_type'],
                'answer':obj['answer'], 'abstained':int(obj['abstained']),
                'human_answerable_in_corpus':'',
                'gold_claims_total':str(len(g.get('gold_claims',[]))) if g.get('answerable') else '',
                'gold_claims_covered':'',
                'gold_evidence_ids':' '.join(g.get('gold_evidence_ids',[])),
                'draft_answerable_in_corpus':str(int(g['answerable'])) if isinstance(g.get('answerable'),bool) else '',
                'gold_annotation_status':g.get('annotation_status',''),
                'gold_answer_draft':g.get('gold_answer',''),
                'gold_claims_draft':' | '.join(g.get('gold_claims',[])),
                'retrieved_evidence_ids':' '.join(x['id'] for x in r['retrieved']),
                'notes':''})
            for i,c in enumerate(obj['claims']):
                claim_rows.append({'question_id':r['question_id'],'group':group,'claim_index':i,
                    'claim':c['text'],'citations':' '.join(c['citations']),
                    'human_factual_correct':'',
                    'human_citation_supported':'' if group!='A' else 'NA',
                    'notes':''})
                for citation in c['citations']:
                    citation_rows.append({'question_id':r['question_id'], 'group':group,
                                          'claim_index':i, 'citation_id':citation,
                                          'human_citation_useful':'', 'notes':''})
    write_csv(directory/'citations_review.csv', citation_rows,
              ['question_id','group','claim_index','citation_id','human_citation_useful','notes'])
    write_csv(directory/'claims_review.csv', claim_rows,
              ['question_id','group','claim_index','claim','citations',
               'human_factual_correct','human_citation_supported','notes'])
    write_csv(directory/'questions_review.csv', q_rows,
              ['question_id','group','question','question_type','answer','abstained',
               'human_answerable_in_corpus','gold_claims_total','gold_claims_covered',
               'gold_evidence_ids','draft_answerable_in_corpus','gold_annotation_status',
               'gold_answer_draft','gold_claims_draft','retrieved_evidence_ids','notes'])
    print(f'Wrote review templates to {directory}. Fill in 1/0, do not modify IDs.')
    print('Draft Gold fields are convenience references, NOT independently verified human labels.')
    print('human_answerable_in_corpus, gold_claims_covered and claim/citation labels remain blank.')
    print('IMPORTANT: evaluate factuality using the original source corpus, NOT the model verifier.')


def _binary(value):
    s = str(value).strip().lower()
    return {'1':1,'0':0,'true':1,'false':0,'yes':1,'no':0}.get(s)


def evaluate_review(claim_csv, question_csv, allow_partial=False, citation_csv=None):
    claims = load_csv(claim_csv)
    questions = load_csv(question_csv)
    citations = load_csv(citation_csv) if citation_csv and Path(citation_csv).exists() else []
    results = {}
    for group in ('A','B','C'):
        qg = [r for r in questions if r['group']==group]
        cg = [r for r in claims if r['group']==group]
        ctg = [r for r in citations if r['group']==group]
        factual_labels = [_binary(r['human_factual_correct']) for r in cg]
        cit_labels = [_binary(r['human_citation_supported']) for r in cg] if group!='A' else []
        annotated_factual = [x for x in factual_labels if x is not None]
        annotated_cit = [x for x in cit_labels if x is not None]
        usefulness = [_binary(r['human_citation_useful']) for r in ctg]
        annotated_useful = [x for x in usefulness if x is not None]
        if not allow_partial and len(annotated_factual)!=len(cg):
            raise ValueError(f'{group}: {len(cg)-len(annotated_factual)} factual claims unlabeled; complete claims_review.csv or pass --allow-partial')
        if not allow_partial and group!='A' and len(annotated_cit)!=len(cg):
            raise ValueError(f'{group}: {len(cg)-len(annotated_cit)} citations unlabeled; complete claims_review.csv or pass --allow-partial')
        if not allow_partial and group!='A' and len(annotated_useful)!=len(ctg):
            raise ValueError(f'{group}: {len(ctg)-len(annotated_useful)} citation-usefulness labels missing; complete citations_review.csv or pass --allow-partial')
        n=len(qg)
        covered=[]
        evidence_recall=[]
        refusal=[]
        for r in qg:
            tot = r['gold_claims_total'].strip()
            got = r['gold_claims_covered'].strip()
            if tot and got:
                try:
                    t, v = int(tot), int(got)
                    if 0 <= v <= t and t>0:
                        covered.append((v,t))
                except ValueError:
                    pass
            gold_ids = set(r.get('gold_evidence_ids','').split())
            retrieved_ids = set(r.get('retrieved_evidence_ids','').split())
            if gold_ids and group!='A':
                evidence_recall.append(len(gold_ids & retrieved_ids)/len(gold_ids))
            answerable = _binary(r['human_answerable_in_corpus'])
            if answerable is not None and not answerable:
                refusal.append(1 if _binary(r['abstained'])==1 else 0)
        results[group] = {
            'questions': n,
            'answer_coverage': sum(_binary(r['abstained'])==0 for r in qg)/n if n else None,
            'factual_precision': sum(annotated_factual)/len(annotated_factual) if annotated_factual else None,
            'factual_labeled':len(annotated_factual),'factual_total':len(cg),
            'citation_support_rate':sum(annotated_cit)/len(annotated_cit) if annotated_cit else None,
            'citation_labeled':len(annotated_cit),
            'citation_usefulness_precision':sum(annotated_useful)/len(annotated_useful) if annotated_useful else None,
            'citation_usefulness_labeled':len(annotated_useful),
            'evidence_recall_at_k':sum(evidence_recall)/len(evidence_recall) if evidence_recall else None,
            'gold_evidence_questions_labeled':len(evidence_recall),
            'gold_claim_recall':sum(v for v,t in covered)/sum(t for v,t in covered) if covered else None,
            'gold_questions_labeled':len(covered),
            'correct_abstention_on_unanswerable':sum(refusal)/len(refusal) if refusal else None,
            'unanswerable_questions_labeled':len(refusal)
        }
        pre, rec = results[group]['factual_precision'], results[group]['gold_claim_recall']
        results[group]['claim_f1'] = 2*pre*rec/(pre+rec) if pre is not None and rec is not None and pre+rec>0 else None
    return results


def save_metrics(stats, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    flat=[dict(group=g,**v) for g,v in stats.items()]
    write_csv(out/'metrics.csv',flat,list(flat[0].keys()))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metrics=['answer_coverage','factual_precision','citation_support_rate',
                 'citation_usefulness_precision','gold_claim_recall','evidence_recall_at_k']
        for m in metrics:
            groups=[g for g in ('A','B','C') if stats[g][m] is not None]
            if not groups:
                continue
            fig, ax=plt.subplots(figsize=(6,4))
            vals=[stats[g][m] for g in groups]
            ax.bar(groups,vals)
            ax.set_ylim(0,1)
            ax.set_ylabel(m.replace('_',' ').title())
            ax.set_title(f'{m} (human labels only)')
            for x,v in enumerate(vals): ax.text(x,v+0.015,f'{v:.1%}',ha='center')
            fig.tight_layout()
            fig.savefig(out/f'{m}.png',dpi=160)
            plt.close(fig)
    except ImportError:
        print('matplotlib not installed; metrics saved without charts')
