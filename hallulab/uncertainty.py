"""Sampling-based diagnostic: SelfCheck-like contradiction and discrete semantic entropy.

These are simplified LLM-judge approximations, NOT exact reproduction of original papers.
"""
import math

COMPARE_PROMPT = '''Compare two candidate answers to the SAME question.
Two answers are equivalent ONLY when they convey the same meaning in BOTH directions
(ignoring harmless paraphrases), and neither adds material facts unsupported by the other.
Set contradictory=true only if they make logically incompatible assertions.
Insufficient/unspecified content is not a contradiction.
Return JSON {"equivalent":true|false,"contradictory":true|false}.'''

CLAIM_COMPARE_PROMPT = '''Does the sampled answer contradict the supplied original atomic claim?
Return JSON {"label":"SUPPORTED|CONTRADICTED|INSUFFICIENT"}. Only output
CONTRADICTED when the sampled answer explicitly conflicts with the claim;
absence of claim in sampled answer is INSUFFICIENT.'''


def pair_check(client, question, x, y):
    r = client.json(COMPARE_PROMPT,
                    f'Question: {question}\nAnswer 1: {x}\nAnswer 2: {y}')
    return bool(r.get('equivalent',False)), bool(r.get('contradictory',False))


def cluster_samples(client, question, answers):
    """Greedy equivalence clustering by a representative; order-sensitive approximation."""
    reps, clusters = [], []
    for answer in answers:
        found = None
        for i, rep in enumerate(reps):
            eq, _ = pair_check(client, question, answer, rep)
            if eq:
                found = i
                break
        if found is None:
            reps.append(answer)
            clusters.append([answer])
        else:
            clusters[found].append(answer)
    return clusters


def discrete_entropy(clusters):
    total = sum(map(len, clusters))
    return -sum((len(c)/total)*math.log(len(c)/total) for c in clusters) if total else 0.


def selfcheck_contradiction(client, original_claims, sampled_answers):
    scores = []
    for claim in original_claims:
        results = []
        for sample in sampled_answers:
            r = client.json(CLAIM_COMPARE_PROMPT,
                            f'Original atomic claim: {claim["text"]}\nSampled answer: {sample}')
            label = str(r.get('label','INSUFFICIENT')).upper()
            # Neutral is not contradiction: explicit contradiction count divided by all samples.
            results.append(1 if label=='CONTRADICTED' else 0)
        scores.append(sum(results)/len(results) if results else 0.)
    return sum(scores)/len(scores) if scores else None
