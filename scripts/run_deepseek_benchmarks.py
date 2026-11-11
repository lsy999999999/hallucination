"""One bounded, inexpensive DeepSeek Flash benchmark run. Never persist the API key."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import getpass
import importlib.util
import json
from pathlib import Path
import sys
import time

spec = importlib.util.spec_from_file_location('course_benchmark', Path(__file__).with_name('run_benchmarks.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
PARENT = b.OUT
b.OUT = b.ROOT / 'results/deepseek_flash_benchmark_20261111'
MODEL = 'deepseek-flash'
BASE_URL = 'https://api.deepseek.com'


def prepare():
    parent_freeze = json.loads((PARENT / 'FREEZE.json').read_text())
    for name, sha in parent_freeze.items():
        assert b.digest((PARENT / name).read_bytes()) == sha
    rag = json.loads((PARENT / 'ragtruth_samples_private.json').read_text())[:60]
    rgb_all = json.loads((PARENT / 'rgb_samples_private.json').read_text())
    base_ids = list(dict.fromkeys(x['sample_id'] for x in rgb_all if x['subset'] == 'en_refine'))[:15]
    fact_ids = list(dict.fromkeys(x['sample_id'] for x in rgb_all if x['subset'] == 'en_fact'))[:10]
    rgb = [x for x in rgb_all if x['sample_id'] in (base_ids if x['subset'] == 'en_refine' else fact_ids)]
    rag_ids = {x['id'] for x in rag}
    rgb_ids = {x['id'] for x in rgb}
    tasks = [t for t in json.loads((PARENT / 'tasks.json').read_text())
             if t['sample_id'] in (rag_ids if t['dataset'] == 'ragtruth' else rgb_ids)]
    assert len(tasks) == 200
    protocol = {
        'version': 'deepseek-course-budget-v1', 'parent_freeze': parent_freeze,
        'selection': 'First 60 RAGTruth, first 15 RGB base and first 10 RGB fact questions in the already frozen random ordering; selected before model calls.',
        'seed': b.SEED, 'ragtruth_n': len(rag), 'rgb_base_n': len(base_ids), 'rgb_fact_n': len(fact_ids),
        'ragtruth_gold_counts': dict(Counter(str(x['gold_hallucination']) for x in rag)),
        'ragtruth_sensitivity': 'Primary: any annotated span. Secondary: exclude implicit_true annotations.',
        'model': MODEL, 'endpoint': BASE_URL, 'thinking': 'disabled', 'temperature': 0,
        'initial_calls': len(tasks), 'verification_max': 65, 'max_attempts_per_item': 2,
        'max_tokens': {'direct': 512, 'claim_check': 1800, 'generate': 512, 'verify': 512},
        'retry_policy': 'At most one retry for a failed/invalid call; on truncation only, double output allowance. No test-driven prompt changes.',
        'stopping_rule': 'Finish this fixed subset and its exact-answer support verification, then stop. No model/sample expansion.',
        'gold_hidden': True, 'tools': False, 'prompts_unchanged_from_parent': True,
        'rgb_scoring': 'Official alias substring matching on final answer; mismatch is not automatically a confirmed factual error.',
        'verification': 'Document-only support check on the exact same generated answer; remove unsupported answers, never regenerate.',
        'limitations': ['small subset, one generation per condition', 'structured prompt adaptation, not full leaderboard reproduction',
                        'public benchmark training exposure unknown', 'source support differs from factual correctness',
                        'same Flash model used for generation and verification', 'no factual gold for every possible true alternative answer'],
        'pricing_source': 'https://api-docs.deepseek.com/quick_start/pricing/',
        'pricing_usd_per_million': {'offpeak': {'input_miss': .15, 'input_hit': .003, 'output': .6},
                                    'peak': {'input_miss': .3, 'input_hit': .006, 'output': 1.2}},
    }
    for name, obj in [('PROTOCOL.json', protocol), ('ragtruth_samples_private.json', rag),
                      ('rgb_samples_private.json', rgb), ('tasks.json', tasks)]:
        b.freeze(b.OUT / name, obj)
    b.freeze(b.OUT / 'FREEZE.json', {name: b.digest((b.OUT / name).read_bytes()) for name in
             ['PROTOCOL.json', 'ragtruth_samples_private.json', 'rgb_samples_private.json', 'tasks.json']})
    print(b.dumps({'initial_calls': len(tasks), 'ragtruth': protocol['ragtruth_gold_counts'], 'out': str(b.OUT)}))


def usage_summary():
    totals = Counter()
    returned_models = Counter()
    failed = 0
    for p in (b.OUT / 'calls').glob('*.json'):
        audit = json.loads(p.read_text())
        for attempt in audit.get('attempts', []):
            raw = attempt.get('raw_response')
            if not raw:
                failed += 1
                continue
            totals['received_responses'] += 1
            returned_models[raw.get('model', 'unknown')] += 1
            u = raw.get('usage', {})
            for k in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens']:
                totals[k] += u.get(k) or 0
    hits = totals['prompt_cache_hit_tokens']
    misses = totals['prompt_tokens'] - hits
    offpeak = (hits * .003 + misses * .15 + totals['completion_tokens'] * .6) / 1e6
    b.write(b.OUT / 'USAGE.json', {'totals': dict(totals), 'returned_models': dict(returned_models),
        'failed_attempts_without_usage': failed, 'cost_offpeak_usd': offpeak,
        'cost_peak_usd': offpeak * 2, 'cost_note': 'Token-based estimate at official rates, not an account billing receipt. Failed requests without usage have unknown billing.'})


def run():
    from openai import OpenAI
    from hallulab.llm import parse_json, safe_error
    for name, sha in json.loads((b.OUT / 'FREEZE.json').read_text()).items():
        assert b.digest((b.OUT / name).read_bytes()) == sha
    b.freeze(b.OUT / 'BACKEND.json', {'model': MODEL, 'endpoint': BASE_URL, 'thinking': 'disabled',
        'temperature': 0, 'workers': 3, 'runner_sha256': b.digest(Path(__file__).read_bytes()),
        'shared_runner_sha256': b.digest(Path(b.__file__).read_bytes()), 'python': sys.version})
    key = getpass.getpass('DeepSeek API key (not saved): ')
    if not key:
        raise ValueError('Missing API key')
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=90, max_retries=0)
    (b.OUT / 'calls').mkdir(exist_ok=True)

    def execute(task):
        path = b.OUT / 'calls' / (task['id'] + '.json')
        if path.exists():
            old = json.loads(path.read_text())
            assert old['task_sha256'] == b.digest(b.dumps(task).encode())
            if old['status'] == 'ok':
                b.validate(task, old['result'])
                return task['id'], 'cached'
        limit = 1800 if task.get('condition') == 'claim_check' else 512
        audit = {'task_id': task['id'], 'task_sha256': b.digest(b.dumps(task).encode()),
            'request': {'system': task['system'], 'input': task['input']}, 'attempts': [],
            'status': 'started', 'started_unix': time.time()}
        b.write(path, audit)
        for attempt in range(2):
            try:
                started = time.time()
                response = client.chat.completions.create(model=MODEL,
                    messages=[{'role': 'system', 'content': task['system'] + '\nOutput valid JSON only.'},
                              {'role': 'user', 'content': b.dumps(task['input'])}],
                    response_format={'type': 'json_object'}, temperature=0, max_tokens=limit,
                    extra_body={'thinking': {'type': 'disabled'}})
                raw = response.model_dump(mode='json')
                audit['attempts'].append({'max_tokens': limit, 'started_unix': started,
                                          'finished_unix': time.time(), 'raw_response': raw})
                b.write(path, audit)
                choice = response.choices[0]
                if choice.finish_reason != 'stop':
                    if choice.finish_reason == 'length':
                        limit *= 2
                    raise ValueError('Incomplete response: ' + str(choice.finish_reason))
                if choice.message.tool_calls:
                    raise ValueError('Unexpected tool call')
                result = parse_json(choice.message.content)
                b.validate(task, result)
                audit.update(status='ok', result=result, finished_unix=time.time())
                b.write(path, audit)
                return task['id'], 'ok'
            except Exception as exc:
                # Never print or save the key, even if a transport includes it in an exception.
                message = safe_error(str(exc).replace(key, '<redacted>'))
                audit['attempts'].append({'error': message, 'finished_unix': time.time()})
                b.write(path, audit)
                if getattr(exc, 'status_code', None) in {400, 401, 402, 403, 404}:
                    break
                if attempt == 0:
                    time.sleep(1)
        audit.update(status='failed', finished_unix=time.time())
        b.write(path, audit)
        raise RuntimeError('Model call failed: ' + task['id'] + '; see sanitized audit.')

    def execute_all(tasks):
        pending = []
        for task in tasks:
            path = b.OUT / 'calls' / (task['id'] + '.json')
            if path.exists() and json.loads(path.read_text()).get('status') == 'ok':
                old = json.loads(path.read_text())
                assert old['task_sha256'] == b.digest(b.dumps(task).encode())
                b.validate(task, old['result'])
            else:
                pending.append(task)
        print('pending', len(pending), 'of', len(tasks), flush=True)
        # A real benchmark item doubles as the connectivity probe; it is never re-run on success.
        if pending:
            print(*execute(pending.pop(0)), flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(execute, task) for task in pending]
            for future in as_completed(futures):
                try:
                    print(*future.result(), flush=True)
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise

    try:
        tasks = json.loads((b.OUT / 'tasks.json').read_text())
        b.write(b.OUT / 'STATUS.json', {'status': 'running', 'started_unix': time.time()})
        execute_all(tasks)
        verify = []
        for task in tasks:
            if task['phase'] != 'generate' or task['condition'] == 'direct':
                continue
            result = json.loads((b.OUT / 'calls' / (task['id'] + '.json')).read_text())['result']
            if result['status'] == 'answer':
                verify.append({'id': 'verify_' + task['id'], 'phase': 'verify', 'system': b.VERIFY,
                    'input': {**task['input'], 'candidate_answer': result['answer']}})
        b.freeze(b.OUT / 'verify_tasks.json', verify)
        execute_all(verify)
        b.score()
        usage_summary()
        b.write(b.OUT / 'STATUS.json', {'status': 'complete', 'initial_calls': len(tasks),
            'verify_calls': len(verify), 'finished_unix': time.time(), 'further_calls': False})
    except Exception as exc:
        usage_summary()
        b.write(b.OUT / 'STATUS.json', {'status': 'failed', 'error': str(exc).replace(key, '<redacted>'),
                                      'finished_unix': time.time()})
        raise RuntimeError(str(exc).replace(key, '<redacted>')) from None
    finally:
        client.close()
        key = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'score'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'score':
        b.score()
        usage_summary()
    else:
        run()
