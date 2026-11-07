import argparse
import hashlib
import json
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl, append_jsonl

DEFAULT_MANIFEST='data/papers_manifest.json'
DEFAULT_PDF='data/papers'
DEFAULT_CORPUS='data/corpus.jsonl'
DEFAULT_QUESTIONS='data/questions.jsonl'
DEFAULT_RESULTS='results/experiment.jsonl'
DEFAULT_GOLD='data/gold_annotations_draft.jsonl'


def make_parser():
    p=argparse.ArgumentParser(prog='python -m hallulab',
        description='LLM hallucination and citation verification mini-study')
    subs=p.add_subparsers(dest='command',required=True)
    a=subs.add_parser('download',help='Download public paper PDFs')
    a.add_argument('--manifest',default=DEFAULT_MANIFEST)
    a.add_argument('--pdf-dir',default=DEFAULT_PDF)
    a=subs.add_parser('ingest',help='Turn PDFs into page-aware JSONL chunks')
    a.add_argument('--manifest',default=DEFAULT_MANIFEST)
    a.add_argument('--pdf-dir',default=DEFAULT_PDF)
    a.add_argument('--out',default=DEFAULT_CORPUS)
    a.add_argument('--chunk-words',type=int,default=210)
    a=subs.add_parser('run',help='Run paired A/B/C experiment; API key required')
    a.add_argument('--corpus',default=DEFAULT_CORPUS)
    a.add_argument('--questions',default=DEFAULT_QUESTIONS)
    a.add_argument('--out',default=DEFAULT_RESULTS)
    a.add_argument('--limit',type=int,default=30)
    a.add_argument('--k',type=int,default=4)
    a.add_argument('--model',default=None)
    a.add_argument('--verifier-model',default=None)
    a.add_argument('--retriever',choices=['bm25','hybrid'],default='bm25')
    a.add_argument('--restart',action='store_true',help='Overwrite prior run, instead of resuming')
    a=subs.add_parser('gold-check',help='Validate Gold IDs and draft status (no API)')
    a.add_argument('--gold',default=DEFAULT_GOLD)
    a.add_argument('--questions',default=DEFAULT_QUESTIONS)
    a.add_argument('--corpus',default=DEFAULT_CORPUS)
    a=subs.add_parser('retrieval-check',help='Recall@k from question-only retrieval (no API)')
    a.add_argument('--gold',default=DEFAULT_GOLD)
    a.add_argument('--questions',default=DEFAULT_QUESTIONS)
    a.add_argument('--corpus',default=DEFAULT_CORPUS)
    a.add_argument('--k',type=int,default=4)
    a.add_argument('--retriever',choices=['bm25','hybrid'],default='bm25')
    a.add_argument('--out-dir',default='results/gold_diagnostics')
    a=subs.add_parser('review',help='Export independent human review forms')
    a.add_argument('--run',default=DEFAULT_RESULTS)
    a.add_argument('--gold',default=DEFAULT_GOLD)
    a.add_argument('--questions',default=DEFAULT_QUESTIONS)
    a.add_argument('--corpus',default=DEFAULT_CORPUS)
    a.add_argument('--out-dir',default='results/review')
    a.add_argument('--overwrite',action='store_true',help='May overwrite prior human annotations')
    a=subs.add_parser('score',help='Compute metrics from human review forms')
    a.add_argument('--review-dir',default='results/review')
    a.add_argument('--out-dir',default='results/metrics')
    a.add_argument('--allow-partial',action='store_true')
    a=subs.add_parser('detect',help='Optional simplified SelfCheck/semantic entropy experiment')
    a.add_argument('--run',default=DEFAULT_RESULTS)
    a.add_argument('--out',default='results/detection.jsonl')
    a.add_argument('--model',default=None)
    a.add_argument('--limit',type=int,default=12)
    a.add_argument('--samples',type=int,default=5)
    a.add_argument('--temperature',type=float,default=1.0)
    a.add_argument('--restart',action='store_true')
    subs.add_parser('smoke',help='Network-free end-to-end contract sanity check')
    return p


def load_client(model):
    from .llm import OpenAIJSONClient
    return OpenAIJSONClient(model=model)


def run_experiment(args):
    from .retrieval import BM25Retriever, HybridRetriever
    from .experiment import run_one
    corpus=read_jsonl(args.corpus)
    if not corpus:
        raise ValueError('Empty corpus: run download and ingest first')
    if args.k <= 0 or args.limit <= 0:
        raise ValueError('Both --k and --limit must be > 0')
    questions=read_jsonl(args.questions)[:args.limit]
    if not questions:
        raise ValueError('No questions found')
    retriever=BM25Retriever(corpus) if args.retriever=='bm25' else HybridRetriever(corpus)
    gen=load_client(args.model)
    ver=load_client(args.verifier_model or args.model)
    settings={'model':gen.model, 'verifier_model':ver.model, 'retriever':args.retriever,
              'corpus_sha256':hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
              'questions_sha256':hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
              'k':args.k, 'corpus':str(Path(args.corpus).resolve()),
              'questions':str(Path(args.questions).resolve())}
    path=Path(args.out)
    meta=path.with_suffix('.meta.json')
    if args.restart:
        path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
    if meta.exists():
        old=json.loads(meta.read_text())
        if old != settings:
            raise ValueError('Run settings changed. Use --restart or a new --out to avoid mixing experiments.')
    else:
        meta.parent.mkdir(parents=True,exist_ok=True)
        meta.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    completed={r['question_id'] for r in read_jsonl(path)} if path.exists() else set()
    for i,q in enumerate(questions,1):
        if q['id'] in completed:
            print(f'[{i}/{len(questions)}] {q["id"]} cached')
            continue
        r=run_one(q,retriever,gen,ver,args.k)
        append_jsonl(path,r)
        print(f'[{i}/{len(questions)}] {q["id"]} A:{len(r["A"]["claims"])} '
              f'B:{len(r["B"]["claims"])} C:{len(r["C"]["claims"])}')
    print(f'Completed: {path}. Next: python -m hallulab review')


def run_detection(args):
    from .experiment import generate
    from .uncertainty import cluster_samples,discrete_entropy,selfcheck_contradiction
    rows=read_jsonl(args.run)[:args.limit]
    if not rows:
        raise ValueError('Run main experiment first')
    client=load_client(args.model)
    outfile=Path(args.out)
    if args.restart:
        outfile.unlink(missing_ok=True)
    done={r['question_id'] for r in read_jsonl(outfile)} if outfile.exists() else set()
    for i,r in enumerate(rows,1):
        if r['question_id'] in done:
            print(f'[{i}] detection cached {r["question_id"]}')
            continue
        main=r['A']
        samples=[generate(client,r['question'],temperature=args.temperature)['answer'] for _ in range(args.samples)]
        answers=[main['answer']]+samples
        clusters=cluster_samples(client,r['question'],answers)
        contradiction=selfcheck_contradiction(client,main['claims'],samples)
        out={'question_id':r['question_id'],'model':client.model,
             'semantic_entropy_discrete_approx':discrete_entropy(clusters),
             'clusters':clusters,'selfcheck_contradiction_approx':contradiction,
             'sampled_answers':samples}
        append_jsonl(outfile,out)
        print(f'[{i}/{len(rows)}] {r["question_id"]}: entropy={out["semantic_entropy_discrete_approx"]:.3f}')
    print('WARNING: exploratory approximation, not an original-paper reproduction.')


def smoke():
    from .retrieval import BM25Retriever
    from .experiment import run_one
    toy=[
      {'id':'toy_p001_c00','paper_id':'toy','page':1,'title':'Invented classroom fixture',
       'text':'The invented paper uses retrieval to obtain external evidence. Its evaluation splits answers into atomic claims.'},
      {'id':'toy_p002_c00','paper_id':'toy','page':2,'title':'Invented classroom fixture',
       'text':'A citation must support a generated assertion. Unsupported claims are filtered.'},
    ]
    class FakeGenerator:
        def json(self, sys, user, temperature=None):
            return {'answer':'The invented paper uses retrieval.', 'abstained':False,
                    'claims':[{'text':'The invented paper uses retrieval.',
                               'citations':['toy_p001_c00'] if 'Sources (ONLY' in user else []}]}
    class FakeVerifier:
        def json(self, sys, user, temperature=None):
            return {'verdicts':[{'claim_id':'0','label':'SUPPORTED','reason':'literal fixture'}]}
    row=run_one({'id':'TOY','type':'single_fact','question':'What does the invented paper use?'},
                BM25Retriever(toy),FakeGenerator(),FakeVerifier())
    assert len(row['B']['claims'])==1 and len(row['C']['claims'])==1
    assert row['A']['claims'][0]['citations']==[]
    print('SMOKE PASS: retrieval -> A/B generation stub -> verification stub -> C filter')
    print('This verifies application contracts only; it is NOT an actual LLM experiment.')


def main(argv=None):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args=make_parser().parse_args(argv)
    if args.command=='download':
        from .corpus import download_papers
        download_papers(args.manifest,args.pdf_dir)
    elif args.command=='ingest':
        from .corpus import build_corpus
        build_corpus(args.manifest,args.pdf_dir,args.out,args.chunk_words)
    elif args.command=='run': run_experiment(args)
    elif args.command=='gold-check':
        from .gold import validate_gold
        summary=validate_gold(args.gold,args.questions,args.corpus)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        if not summary['valid']:
            raise SystemExit(2)
    elif args.command=='retrieval-check':
        from .gold import retrieval_diagnostic
        summary=retrieval_diagnostic(args.gold,args.questions,args.corpus,args.out_dir,
                                     args.k,args.retriever)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    elif args.command=='review':
        from .evaluation import export_review
        from .gold import validate_gold
        summary=validate_gold(args.gold,args.questions,args.corpus)
        if not summary['valid']:
            raise ValueError('Gold validation errors: '+repr(summary['errors']))
        export_review(args.run,args.out_dir,gold_path=args.gold,overwrite=args.overwrite)
    elif args.command=='score':
        from .evaluation import evaluate_review,save_metrics
        directory=Path(args.review_dir)
        stats=evaluate_review(directory/'claims_review.csv',directory/'questions_review.csv',
                              args.allow_partial, directory/'citations_review.csv')
        save_metrics(stats,args.out_dir)
        print(json.dumps(stats,ensure_ascii=False,indent=2))
    elif args.command=='detect': run_detection(args)
    elif args.command=='smoke': smoke()

if __name__=='__main__': main()
