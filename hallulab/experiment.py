"""Paired groups A (direct), B (RAG), C (deterministic evidence filter)."""
import re

GEN_INSTRUCTIONS = '''You answer scientific questions concisely and factually.
If you cannot answer sufficiently, set abstained=true and return no claims.
Split any substantive answer into 1-5 ATOMIC independently checkable facts.
No combined claims; keep each fact one sentence.
Return JSON: {"answer":"...", "abstained":false, "claims":[{"text":"one atomic fact", "citations":[]}]}
Never invent source identifiers. When sources are supplied, each factual claim must cite
one or more exact provided identifiers (or [] if none truly support it).
When no sources are supplied, ALWAYS use citations=[] for every claim.
Avoid opening new factual assertions in the answer that are not covered by claims.
No sources: answer from your model knowledge, not made-up references.'''

VERIFY_INSTRUCTIONS = '''You are an evidence entailment verifier, not a knowledge expert.
Determine whether ONLY the supplied cited text entails each atomic claim.
Do not use external knowledge. The claim may be true in reality but unsupported by cited text.
SUPPORTED: cited text clearly entails the entire claim.
CONTRADICTED: cited text directly negates the claim.
INSUFFICIENT: cited text is missing, unrelated, ambiguous, or not conclusive.
Numerical values and qualifiers must match precisely.
Return JSON {"verdicts":[{"claim_id":"0","label":"SUPPORTED|CONTRADICTED|INSUFFICIENT","reason":"brief"}]}.
Return exactly one verdict for each claim_id. Do not invent facts.'''


def clean_generated(obj, valid_citation_ids=None):
    claims = []
    valid = None if valid_citation_ids is None else set(valid_citation_ids)
    for c in obj.get('claims', []):
        if not isinstance(c, dict):
            continue
        text = str(c.get('text','')).strip()
        if not text:
            continue
        ids = c.get('citations', [])
        if not isinstance(ids, list):
            ids = []
        ids = list(dict.fromkeys(str(s) for s in ids))
        # INVALID ids stay in output for hallucinated-citation auditing; verifier does not trust them.
        claims.append({'text':text, 'citations':ids})
    abstained = bool(obj.get('abstained',False))
    if abstained:
        claims = []
    if not claims:
        abstained = True
    return {'answer': str(obj.get('answer','')).strip() if not abstained else 'INSUFFICIENT EVIDENCE',
            'abstained':abstained, 'claims':claims}


def generate(client, question, sources=None, temperature=None):
    sources = sources or []
    if sources:
        context = '\n\n'.join(f"[ID: {x['id']} | {x['title']} | PDF page {x['page']}]\n{x['text']}" for x in sources)
        request = f'Question: {question}\n\nSources (ONLY these may be cited):\n{context}\n\nProduce an evidence-grounded answer. If sources cannot support the answer, abstain.'
    else:
        request = f'Question: {question}\n\nNo sources available. Give your own best factual answer or abstain if uncertain.'
    out = clean_generated(client.json(GEN_INSTRUCTIONS, request, temperature=temperature))
    if not sources:
        # A has NO external references by design; prevent hallucinated citation IDs.
        for c in out['claims']:
            c['citations'] = []
    return out


def verify_claims(client, generated, sources):
    byid = {s['id']:s for s in sources}
    claims = generated.get('claims', [])
    if not claims:
        return []
    jobs, verdicts = [], [None] * len(claims)
    for i, c in enumerate(claims):
        cited = list(dict.fromkeys(c.get('citations',[])))
        if not cited or any(cid not in byid for cid in cited):
            verdicts[i] = {'claim_id':str(i),'label':'INSUFFICIENT',
                           'reason':'No citations or fabricated/unretrieved source ID'}
            continue
        evidence = '\n\n'.join(f"[{cid}] {byid[cid]['text']}" for cid in cited)
        jobs.append({'claim_id':str(i),'claim':c['text'], 'cited_evidence':evidence})
    if jobs:
        import json
        out = client.json(VERIFY_INSTRUCTIONS, json.dumps({'items':jobs}, ensure_ascii=False))
        responses = {str(v.get('claim_id')):v for v in out.get('verdicts',[])
                     if isinstance(v,dict) and str(v.get('label','')).upper() in
                     {'SUPPORTED','CONTRADICTED','INSUFFICIENT'}}
        for job in jobs:
            i = int(job['claim_id'])
            v = responses.get(job['claim_id'])
            verdicts[i] = {'claim_id':job['claim_id'],
                           'label':str(v['label']).upper() if v else 'INSUFFICIENT',
                           'reason':str(v.get('reason','')) if v else 'Missing verifier response'}
    return verdicts


def filter_supported(generated, verdicts):
    """C is *filter-only* post-processing, not trained Self-RAG or a repair model."""
    claims = [c for c, v in zip(generated.get('claims',[]), verdicts) if v['label']=='SUPPORTED']
    if not claims:
        return {'answer':'INSUFFICIENT EVIDENCE', 'abstained':True,'claims':[]}
    answer = ' '.join(c['text'] + ' ' + ' '.join(f"[{z}]" for z in c['citations']) for c in claims)
    return {'answer':answer.strip(), 'abstained':False, 'claims':claims}


def run_one(question, retriever, generator_client, verifier_client, top_k=4):
    a = generate(generator_client, question['question'])
    sources = retriever.search(question['question'], top_k=top_k)
    b = generate(generator_client, question['question'], sources=sources)
    verdicts = verify_claims(verifier_client, b, sources)
    c = filter_supported(b, verdicts)
    return {
        'question_id':question['id'], 'question_type':question['type'],
        'question':question['question'], 'retrieved':[
            {'id':x['id'], 'paper_id':x['paper_id'],'page':x['page'],
             'score':x['score']} for x in sources],
        'A':a,'B':b,'C':c,'verdicts_B':verdicts,
        'b_verifier_support_rate':sum(v['label']=='SUPPORTED' for v in verdicts)/len(verdicts) if verdicts else None
    }
