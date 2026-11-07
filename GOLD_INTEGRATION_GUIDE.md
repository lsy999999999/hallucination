# v1.2 — Gold 接入指南（Mac / VS Code）

这不是全新的工程；安装这个补丁**不会替换你的 .env、已有下载的 PDF、data/corpus.jsonl、data/questions.jsonl 或实验结果**。

## A. 安装补丁

下载 `hallulab_gold_v1_2_patch.zip` 到 Mac 的 Downloads 目录，进入原项目：

```bash
cd /Applications/deep_learning_usage/hallucination_midterm
cp hallulab/cli.py hallulab/cli.py.bak
cp hallulab/evaluation.py hallulab/evaluation.py.bak
unzip -o ~/Downloads/hallulab_gold_v1_2_patch.zip -d .
```

补丁包括：
- `hallulab/gold.py`：Gold 校验 / 检索诊断；
- `hallulab/cli.py`：新增 `gold-check`、`retrieval-check`，升级 `review` 命令；
- `hallulab/evaluation.py`：Gold 预填人工评估表，并防止默认覆盖已有评估；
- `data/gold_annotations_draft.jsonl`：30 道题的**草稿** Gold 标注；
- `tests/test_gold_integration.py`：离线测试；
- `GOLD_INTEGRATION_GUIDE.md`：本说明。

请**不要**将 Gold 文件覆盖 `data/questions.jsonl`！只有评估阶段使用 Gold，不让模型看答案。

## B. 校验（不消耗 API）

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m hallulab gold-check
```

目标输出应有 `valid: true`、`gold_questions: 30`、`corpus_chunks: 558`、`corpus_papers: 8`。

`provisional: 30` 表示所有 Gold 记录仍需要独立人工复核，**不是代码报错**。

## C. 诊断 RAG 是否取回 Gold 证据（不消耗 API）

```bash
python -m hallulab retrieval-check --k 4 --retriever bm25
```

输出：
- `results/gold_diagnostics/retrieval_summary.json`
- `results/gold_diagnostics/retrieval_per_question.csv`

`retrieval_per_question.csv` 给出逐题 `retrieved_ids`、`gold_ids` 和 `missed_gold_ids`。只有标记为可回答的题纳入召回率分母。

**已在用户提供的 558 片段语料和草稿 Gold 上测得**：BM25 Top-4 的 micro evidence recall = 14.75%（9/61），至少命中一个 Gold Evidence ID 的可回答问题比例 = 37.5%（9/24）；Top-8 的对应值为 21.31%（13/61）、54.17%（13/24）。这些是**草稿标注 ID 精确匹配的检索诊断**，不代表事实准确率，也会漏掉语义等价的替代证据。

## D. 运行三组实验（此步会消耗 API）

先用 3 道题测试（如果之前已跑成功，可跳过）：

```bash
python -m hallulab run --limit 3 --out results/pilot.jsonl
```

运行完整 30 题：

```bash
python -m hallulab run --limit 30 --out results/experiment.jsonl
```

注意：A/B/C **不会读取 Gold**，不造成答案泄漏。使用同一 `--out` 会断点续跑；改变检索器或模型需用不同输出路径。

## E. 生成已预填 Gold 的人工核验表（不消耗 API）

```bash
python -m hallulab review --run results/experiment.jsonl \
  --gold data/gold_annotations_draft.jsonl --out-dir results/review_gold
```

打开生成的三个文件：

1. `questions_review.csv`：Gold 草稿信息已经预填，`draft_answerable_in_corpus`、`gold_answer_draft`、`gold_claims_draft`、`gold_evidence_ids` 仅是帮助人工复核的参考。必须人工填写 `human_answerable_in_corpus`（1/0）、`gold_claims_covered`（0 到必要事实总数）。`gold_claims_total` 由草稿预填，但复核后可以修正。
2. `claims_review.csv`：逐条填写 `human_factual_correct`（1/0）。B/C 还需逐条填写 `human_citation_supported`（1/0），只凭该条实际引用的文本判定，而不能照抄 LLM 验证器标签。
3. `citations_review.csv`：逐引用填写 `human_citation_useful`（1/0）。

本次更新特意让 `human_*` 字段保持空白：**草稿 Gold 不是人工确认**。已经存在 `results/review_gold/*.csv` 时，默认拒绝覆盖，保护人工工作；必要时使用一个新的 `--out-dir`，或明确传 `--overwrite`。

## F. 评分和绘图（完成独立人工核验后）

```bash
python -m hallulab score --review-dir results/review_gold --out-dir results/metrics_gold
```

会输出 `metrics.json`、`metrics.csv` 和可视化图片。人工标签不全会报错；探索性测试可暂时传 `--allow-partial`，但此时不能把部分标签的分数称为全样本结论。

### 为什么要先检查 Recall@4？

此处使用原来的 BM25 和问题原文做检索，Top-4 只能命中 9/61 个草稿证据 ID。多文档问题尤其薄弱。建议将这一点作为单独的检索瓶颈分析，后续可在 *不改变 Gold* 的前提下对比改进后的检索器，并明确区分「没有检索到证据」和「有证据但生成仍出错」。

**项目的边界**：当前 Gold 均为 `source_checked_draft_needs_independent_review`，请在正式论文/课堂汇报里明确写作“预标注数据”而不是“人工金标准”。
