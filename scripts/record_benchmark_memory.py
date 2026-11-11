"""Record this user-requested research update as an additive memory note only."""
from datetime import datetime, timezone
from pathlib import Path

note = '''# HalluLab RAGTruth/RGB Flash course benchmark completed

2026-11-11. User replaced the account backend with explicitly authorized official DeepSeek Flash, requested minimal API use and stopping once the course experiment was sufficient. Never retain supplied credentials.

Completed /Applications/deep_learning_usage/hallucination_midterm/results/deepseek_flash_benchmark_20261111. Official data repositories pinned with byte hashes. Before inference, reduced the frozen random ordering to 60 distinct-source RAGTruth QA/test good responses (15 annotated positive,45 negative),15 RGB en_refine questions in four paired evidence conditions,10 en_fact questions in paired original/tampered conditions. deepseek-flash, thinking disabled,temperature0,no tools. 200 initial+53 verification successful responses,3 timeouts retried:256 request attempts. 204413 input+26025 output tokens;3968 cache hits/200445 misses. Official offpeak estimate USD0.045693654;timeout usage/billing unknown. Exact backend weights beyond alias unverified.

RAGTruth official-label direct detection P/R/F1=.667/.667/.667,TP10 FP5 FN5 TN40. Claim-check P/R/F1=.55/.733/.629,TP11 FP9 FN4 TN36. Excluding implicit_true spans leaves12 positives and F1 .593/.563. Preserve official labels; some source/annotation disagreements were documented, not relabeled.

RGB official alias matching versus explicit AI-assisted format review are separate. Direct raw10/15 becomes11/15 after recognizing Tomljanovic diacritic;clean and mixed15/15. Direct has0 confirmed factual errors and4 abstentions, so clean evidence primarily improves coverage, not demonstrated hallucination-rate reduction. Official-negative group raw2/15 becomes3/15 after numeric-date recognition:3correct1wrong11abstain. Negative documents contain answer-bearing fragments, so rejection rate is not independently confirmed correct-rejection rate.

Tampered evidence:3correct6wrong1abstain. Document-only support verifier retains all6wrong and removes all3correct:0correct6wrong4abstain. Instagram example: correct Facebook answer removed because documents falsely say Apple. Shape of Water: wrong Emma Stone answer supported by false documents. Mixed group also loses1correct France runner-up answer despite verifier explanation supporting it. Support and factual truth must remain separate.

metrics.json and original outputs unchanged;metrics_reviewed.json contains four transparent corrections for two answer-format aliases and two fake-answer mention/adoption cases (negated Apple and United States/U.S.). REPORT_CN.md has methods,tables,limits,cases,and five-minute outline. AUDIT.json validates253 outputs,requests,blind inputs,usage and hashes;RESULTS_FREEZE.json freezes artifacts. Five benchmark software checks pass;three older CLI/backend tests had pre-existing failures and were not repaired. Prior experiments and PPT untouched. Model process ended;no further calls after completion. Key read through hidden terminal input only,not persisted in artifacts,memory,or global settings.
'''

if __name__ == '__main__':
    directory = Path('/Users/LSY/.codex/memories/extensions/ad_hoc/notes')
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = directory / f'{stamp}-hallulab-ragtruth-rgb-flash.md'
    with target.open('x') as stream:
        stream.write(note)
    print(target)
