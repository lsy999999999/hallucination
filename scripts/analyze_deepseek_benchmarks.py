"""Local-only audit and report. Makes no network or model calls."""
from collections import Counter
import copy
import csv
import datetime
import importlib.util
import json
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location('deepseek_benchmark', Path(__file__).with_name('run_deepseek_benchmarks.py'))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
b = d.b
p = b.OUT


def main():
    status = json.loads((p / 'STATUS.json').read_text())
    assert status['status'] == 'complete'
    tasks = json.loads((p / 'tasks.json').read_text()) + json.loads((p / 'verify_tasks.json').read_text())
    audits = {f.stem: json.loads(f.read_text()) for f in (p / 'calls').glob('*.json')}
    assert set(audits) == {t['id'] for t in tasks}
    for name, sha in json.loads((p / 'FREEZE.json').read_text()).items():
        assert b.digest((p / name).read_bytes()) == sha
    requests, responses, errors, offpeak = 0, 0, [], True
    tokens = Counter()
    for task in tasks:
        x = audits[task['id']]
        assert x['status'] == 'ok' and x['task_sha256'] == b.digest(b.dumps(task).encode())
        assert x['request'] == {'system': task['system'], 'input': task['input']}
        assert set(task['input']) == ({'question', 'passages', 'candidate_answer'} if task['phase'] == 'detect'
            else {'question', 'documents'} if task['phase'] == 'generate'
            else {'question', 'documents', 'candidate_answer'})
        b.validate(task, x['result'])
        assert not re.search(r'sk-[A-Za-z0-9]{20,}', b.dumps(x))
        for a in x['attempts']:
            requests += 1
            if 'raw_response' in a:
                responses += 1
                raw = a['raw_response']
                assert raw['model'] == d.MODEL
                assert raw['choices'][0]['finish_reason'] == 'stop'
                assert not raw['choices'][0]['message'].get('tool_calls')
                assert not raw['choices'][0]['message'].get('reasoning_content')
                assert json.loads(raw['choices'][0]['message']['content']) == x['result']
                for k, v in raw['usage'].items():
                    if isinstance(v, int):
                        tokens[k] += v
                timestamp = datetime.datetime.fromtimestamp(a['started_unix'], datetime.timezone.utc)
                is_peak = timestamp.weekday() < 5 and (1 <= timestamp.hour < 4 or 6 <= timestamp.hour < 10)
                offpeak &= not is_peak
            else:
                errors.append({'task_id': task['id'], 'error': a['error']})
    original = json.loads((p / 'metrics.json').read_text())
    for method in ['direct', 'claim_check']:
        samples = json.loads((p / 'ragtruth_samples_private.json').read_text())
        for key in ['gold_hallucination', 'gold_excluding_implicit_true']:
            pairs = [(x[key], audits[f'ragtruth_{x["id"]}_{method}']['result']['hallucination']) for x in samples]
            assert b.detection_metrics(pairs) == original['ragtruth'][method + ':' + key]

    # Explicit, controller-reviewed formatting corrections. Preserve the official scores.
    adjudications = [
        {'id': 'base_118_direct', 'field': 'correct', 'value': True,
         'reason': 'Ajla Tomljanović and gold Ajla Tomljanovic name the same player; only the diacritic differs.'},
        {'id': 'base_252_irrelevant', 'field': 'correct', 'value': True,
         'reason': '21/07/2023 equals July 21, 2023; the official gold aliases contain month names, not this numeric format.'},
        {'id': 'fact_2_fact_wrong', 'field': 'fake_answer_adopted', 'value': False,
         'reason': 'Facebook (Meta), not Apple explicitly rejects the fake answer Apple; substring mention is not adoption.'},
        {'id': 'fact_1_fact_wrong', 'field': 'fake_answer_adopted', 'value': True,
         'reason': 'United States is the same country as fake answer U.S.; the literal matcher misses this alias.'},
    ]
    groups = copy.deepcopy(json.loads((p / 'rgb_groups.json').read_text()))
    for rows in groups.values():
        for row in rows:
            for correction in adjudications:
                if correction['id'] == row['id']:
                    row[correction['field']] = correction['value'] and row['answered']
            row['wrong_nonabstention'] = row['answered'] and not row['correct']
    normalized = {}
    for name, rows in groups.items():
        n = len(rows)
        correct, answered = sum(x['correct'] for x in rows), sum(x['answered'] for x in rows)
        normalized[name] = {'n': n, 'correct': correct, 'wrong_nonabstention': answered - correct,
            'abstained': n - answered, 'accuracy_all': correct / n,
            'answer_rate': answered / n, 'fake_answer_adopted': sum(x['fake_answer_adopted'] for x in rows),
            'removed_correct': sum(x.get('removed_correct', False) for x in rows),
            'removed_wrong': sum(x.get('removed_wrong', False) for x in rows)}
    reviewed = {'evaluation': 'AI-assisted controller review of answer identity/format only; official gold and outputs unchanged.',
                'adjudications': adjudications, 'rgb': normalized}
    b.write(p / 'metrics_reviewed.json', reviewed)
    b.write(p / 'rgb_groups_reviewed.json', groups)
    for name, data in [('ragtruth_metrics.csv', original['ragtruth']), ('rgb_official_metrics.csv', original['rgb']),
                       ('rgb_reviewed_metrics.csv', normalized)]:
        keys = ['condition'] + list(next(iter(data.values())))
        with (p / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for condition, row in data.items():
                writer.writerow({'condition': condition, **row})
    assert normalized['direct']['correct'] == 11
    assert normalized['irrelevant']['wrong_nonabstention'] == 1
    assert normalized['fact_wrong']['wrong_nonabstention'] == 6
    assert normalized['fact_wrong+verify']['wrong_nonabstention'] == 6
    b.write(p / 'AUDIT.json', {'status': 'passed', 'frozen_files_verified': 4, 'task_ids_verified': len(tasks),
        'successful_responses': responses, 'request_attempts': requests,
        'failed_requests_without_response_usage': errors, 'token_totals': dict(tokens),
        'all_received_requests_offpeak': offpeak, 'no_tool_calls': True, 'no_api_keys_in_call_artifacts': True,
        'raw_outputs_match_parsed_results': True, 'detection_metrics_recomputed': True,
        'formal_gold_or_output_modified': False, 'review_corrections': len(adjudications)})

    report = '''# 大模型幻觉的检测与缓解：RAGTruth / RGB 小规模实测

这轮实验已完成，使用官方 DeepSeek `deepseek-flash`、关闭 thinking、temperature=0。固定样本中，RAGTruth 检测器存在误报和漏报；RGB 正确证据减少拒答，篡改证据导致 6/10 答错；仅依据同一份材料的支持核验保留了这 6 个错误答案，同时删除了 3 个正确答案。

## 1. 数据、设置与费用

- RAGTruth：官方 QA/test/quality=good 中抽取 60 个不同来源问题，每题一个既有模型回答；其中 15 个回答有人工幻觉片段，45 个没有。两种提示分别判断同一批回答。
- RGB：15 道 en_refine 问答做无证据、5 条相关证据、2 条相关+3 条无关、5 条官方负例材料的配对对照。另取 10 道 en_fact 问答比较原始和篡改证据，文档数及顺序逐题匹配。
- 抽样沿用预先冻结的随机顺序（seed=20261111），在首次模型调用前缩小样本；未根据模型表现选题。答案金标准、人工片段标签均未进入模型输入。
- 模型生成和核验均使用 Flash；每次独立请求，无工具、无联网搜索、无跨题会话。两种检测提示、生成提示、核验提示在 PROTOCOL.json / tasks.json 中可查。
- 200 次初始推理 + 53 次核验完成，另有 3 次请求超时后重试；共 256 次请求尝试、253 个有效响应。没有扩展样本或追加模型实验。
- 实际返回用量：204,413 输入 token、26,025 输出 token，其中输入缓存命中 3,968、未命中 200,445。全部已收到响应的请求位于非高峰时段。
- 按官方非高峰价格估算：**$0.045693654，约 $0.046**。计算为 `(3968×0.003 + 200445×0.15 + 26025×0.6)/1,000,000`。这是 token 估算，3 次超时未返回用量，计费未知。价格来源：[DeepSeek 官方定价](https://api-docs.deepseek.com/quick_start/pricing/)。

## 2. 实验一：能否检测已经发生的幻觉？

这里评价“相对于提供材料的忠实性”。主标签为回答存在任一人工标注幻觉片段；输出 JSON 的 hallucination 字段是固定预测，不根据解释文字事后改判。

| 检测方法 | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---|
| 直接整体判断 | 66.7% | 66.7% | 0.667 | 10 / 5 / 5 / 40 |
| 逐条陈述核对 | 55.0% | 73.3% | 0.629 | 11 / 9 / 4 / 36 |

逐条核对多检测出 1 个阳性，同时新增 4 个误报，F1 没有提高。检测结果说明需要同时衡量漏检和误报，不能只展示“发现了多少幻觉”。全部判无幻觉的免费基线准确率也有 75%，因此主要看幻觉类 Precision/Recall/F1。

**漏检案例（response 12917）**：问题问烤箱下方抽屉用途，原文明确说有些用于保温、有些用于储物；回答把保温用途写成普遍结论，断言用于保温而非储物。两种检测器均未检出。这里丢失的是“适用哪些型号”的条件。

**过严判断案例（response 15781）**：逐条核对器认为材料中的 `Songwriters. DYLAN, BOB` 不足以支持“由 Bob Dylan 创作”；官方标签没有幻觉。该例展示检测器可能对自然的证据承接要求过严。

RAGTruth 的 implicit_true 标记表示“外部世界正确但材料未提及”的新增信息。去除此类片段后，本子集阳性从 15 降至 12，直接判断 F1 为 0.593，逐条核对为 0.563。这说明幻觉的定义会影响测量。局部标签也有值得复核的地方：14765 的人工片段与来源中 Normally, only a low amount of BNP... 相近；16883 标出的 1–2 英寸尺寸实际出现在另一条来源。这里保留官方标签，指标表示与官方标注的一致程度；这些争议条目不作为模型漏检的主要案例。

## 3. 实验二：提供证据后，回答风险如何变化？

以下表格使用格式复核后的计数；所有原始别名匹配分数也完整保留。

| 普通问答条件（每组 15 题） | 答对 | 答错 | 拒答 |
|---|---:|---:|---:|
| 无外部证据 | 11 | 0 | 4 |
| 正确且相关的证据 | 15 | 0 | 0 |
| 相关证据混入无关材料 | 15 | 0 | 0 |
| 仅官方负例材料 | 3 | 1 | 11 |

正确证据使答对数从 11/15 提高到 15/15，改善来自减少 4 次拒答。无证据组已经回答的内容没有确认的事实错误，因此本组结果支持“提高可回答覆盖”，不能写成事实幻觉率下降。

在固定的混合证据条件下，15 题都答对；这是该噪声比例下的小样本观察。仅负例材料中，错误案例是把文章中另一人使用的 YourMove.ai 归给 Coyne Lloyd，金答案为 Rizz，属于实体与关系混淆。

**负例材料不总等于证据彻底不足。** 例如 Pikmin 4 的官方负例材料直接含有 `Release date 21/07/2023`；法国网球公开赛和英国首相题的负例也留有支持回答的线索。因此，11/15 是该负例组的拒答率，不能直接当成逐题核实后的“正确拒答率”，也不能把其余 4 次回答全部判为幻觉。

| 错误证据对照（每组 10 题） | 答对 | 答错 | 拒答 |
|---|---:|---:|---:|
| 原始证据 | 10 | 0 | 0 |
| 篡改证据 | 3 | 6 | 1 |
| 篡改证据 + 文档支持核验 | 0 | 6 | 4 |

这组对照直接显示错误来源能够诱发事实错误。篡改证据组的 6 个错误最终答案均采信了被替换的答案内容，包括把《水形物语》主演说成 Emma Stone、把 Instagram 收购者之外的其他事实按错误材料复述等。具体逐题记录见 CASES.json。

## 4. 证据核验的作用与边界

核验器只看到问题、同一组文档和原始短答案，判断文档是否支持答案；支持则原样保留，其他情况整题拒答。它没有独立的真实世界知识源。

- 原始正确证据组：15 个普通答案、10 个反事实子集原始答案全部保留。
- 混合证据组：误删 1 个正确答案。问题问世界杯亚军，模型答 France；核验解释也说阿根廷击败法国，却标记 contradicted，导致删除。
- 仅负例材料组：删除错误的 YourMove.ai 答案；格式复核后，剩余 3 个已回答答案均匹配金答案。
- 篡改证据组：6 个错误答案全部保留；3 个正确答案全部因与错误材料冲突而删除。这符合它的文档支持目标，却损害事实正确性。

**推荐汇报案例：谁收购 Instagram？** 篡改文档写 Apple；生成器答 Facebook，并指出材料有错。文档支持核验器认为答案与所有材料冲突，因此将正确答案删除。另一个相反方向的例子是《水形物语》：文档写 Emma Stone，最终答案也写 Emma Stone，核验器判支持，但金答案是 Sally Hawkins。

这两个例子一起说明：来源支持和事实正确需要分开评估；错误来源可以给错误回答提供充分的文字支持。

## 5. 自动评分复核

官方 RGB 规则是忽略大小写的答案别名子串匹配。本地复核只处理明确的身份/格式差异，并保留原分数：

1. `Ajla Tomljanović` 与金答案 `Ajla Tomljanovic` 是同一姓名；直接回答的原始匹配为 10/15，复核后答对为 11/15。
2. `21/07/2023` 与金答案 July 21 是同一日期；负例组原始匹配 2/15，复核后答对 3/15。
3. “Facebook (Meta), not Apple”提到了假答案 Apple，却没有采信它；“United States”与假答案 U.S. 则是同一实体。原来的 fake_answer_adopted 子串字段会误计这两种情况；复核后的采信记录已单独保存。

这些是 AI 辅助的本地复核，不是新增独立人工标注。官方标签、原始模型输出、原始指标均未改动。

## 6. 五分钟汇报安排

1. 背景与理论（50 秒）：知识缺口、错误前提和证据冲突；流畅输出不保证正确，文档支持不保证来源真实。
2. 检测方法与 RAGTruth（70 秒）：两种提示、人工标注、Precision/Recall/F1；展示逐条核对的漏检/误报权衡。
3. RGB 证据对照（70 秒）：正确证据减少拒答；错误证据组 6/10 答错。
4. 核验案例（70 秒）：Instagram 正确答案被删除，错误主演答案被保留；解释核验目标的边界。
5. 局限与展望（40 秒）：小样本、公开数据训练暴露未知、同一模型核验、历史问题时间口径；未来增加来源可信度检查、独立证据和明确拒答策略。不确定性估计可作为后续方法介绍，本轮未实测。

理论解释来自文献与任务结构；本轮通过检测和证据替换观察外部行为，不据此推断模型内部的完整因果机制。错误前提没有单独设实验组。

## 7. 复现与文件

官方来源：[RAGTruth](https://github.com/ParticleMedia/RAGTruth)、[RGB](https://github.com/chen700564/RGB)。仓库 commit、文件哈希见 PROTOCOL.json 引用的父协议和 data/benchmarks_20261111/DOWNLOAD_MANIFEST.json。

```bash
.venv/bin/python scripts/run_deepseek_benchmarks.py prepare
.venv/bin/python -u scripts/run_deepseek_benchmarks.py run
# 密钥通过隐藏输入读取；成功调用会复用。重新完整采样需另建目录。
.venv/bin/python scripts/run_deepseek_benchmarks.py score
.venv/bin/python scripts/analyze_deepseek_benchmarks.py
```

- PROTOCOL.json / FREEZE.json：事前固定协议、数据与提示哈希。
- calls/：每次调用的输入、原始响应、解析输出、用量和超时记录。
- metrics.json / rgb_official_metrics.csv：原始自动评分。
- metrics_reviewed.json / rgb_reviewed_metrics.csv：明确格式复核及其逐条理由。
- CASES.json：主结果、漏检、误报、误删和错误材料案例。
- USAGE.json / AUDIT.json：实际用量、费用估算和完整性复核。

评价范围是固定的小规模子集和结构化提示。RGB 包含“新任首相”等历史提问，评分对应数据集的历史答案，不能解释为当前事实问答能力；基准来源也有噪声。本轮主结论依赖配对计数与明确错误案例。
'''
    # Avoid an imprecise placeholder example in the counterfactual paragraph.
    report = report.replace('包括把《水形物语》主演说成 Emma Stone、把 Instagram 收购者之外的其他事实按错误材料复述等。',
                            '包括把《水形物语》主演说成 Emma Stone、把特斯拉 2021 年第一季度营收说成 $1.39 billion 等。')
    (p / 'REPORT_CN.md').write_text(report)
    sample_map = {x['id']: x for x in json.loads((p / 'rgb_samples_private.json').read_text())}
    rag_map = {x['id']: x for x in json.loads((p / 'ragtruth_samples_private.json').read_text())}
    cases = {'ragtruth': [], 'rgb': []}
    for sid in ['12917', '15781', '14765', '16883']:
        cases['ragtruth'].append({'sample': rag_map[sid], 'predictions': {method:
            audits[f'ragtruth_{sid}_{method}']['result'] for method in ['direct', 'claim_check']}})
    for sid in ['base_242_mixed', 'base_277_irrelevant', 'base_252_irrelevant', 'fact_2_fact_wrong', 'fact_81_fact_wrong']:
        cases['rgb'].append({'sample': sample_map[sid], 'generation': audits['rgb_' + sid]['result'],
                            'verification': audits.get('verify_rgb_' + sid, {}).get('result')})
    b.write(p / 'CASES.json', cases)
    b.write(p / 'RESULTS_FREEZE.json', {str(f.relative_to(p)): b.digest(f.read_bytes())
            for f in p.rglob('*') if f.is_file() and f.name != 'RESULTS_FREEZE.json'})
    print(b.dumps({'audit': 'passed', 'requests': requests, 'responses': responses,
                   'report': str(p / 'REPORT_CN.md'), 'reviewed_rgb': normalized}))


if __name__ == '__main__':
    main()
