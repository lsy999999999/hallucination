# 大模型幻觉检测、RAG 与引用验证：期中课程实验

> 实现的是小型、可复现的**探索性实验**，不是 Self-RAG / ALCE / SelfCheckGPT 的完全复现。实验结果需要人工标注后才能作为正式结论。

## 研究问题与分组

- **A / Direct LLM**: 无检索，直接回答；不应输出有效引用。
- **B / RAG**: BM25 检索 PDF 原文前 k 个片段，生成 1–5 条原子事实，每条绑定引用 ID。
- **C / RAG + Verify**: **使用 B 组同一份生成结果**，逐条核对 B 的引用证据；只保留证据蕴含的事实，其他事实删除。C 是过滤器，不是完整 Self-RAG，也不意味着所有留下来的事实都真实。
- **可选检测实验**: 对 A 生成额外样本，运行简化的 SelfCheck 风格矛盾率和离散语义熵（LLM 等价判断，近似而非原论文实现）。

## 最低要求

- macOS / Linux / Windows, Python 3.10+，推荐 Python 3.11。
- 正式实验可使用 ChatGPT 账号登录后的 Codex CLI，或能调用 Responses API 的 API Key。Codex 使用账号的 Codex 可用额度；**网页聊天额度、Codex 限额和 API 余额需分别核对**。
- `download` 需要互联网；论文网站若阻断自动下载，请手工下载 PDF 并按照 `data/papers_manifest.json` 的 id 命名存放到 `data/papers/`。
- `smoke` 不需要网络或 API Key，它只是检查代码路径是否能连通，不能当真实实验结果。

## macOS 从零运行

```bash
cd hallucination_midterm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m hallulab smoke
python -m unittest discover -s tests -v
```

接着准备论文与知识库：

```bash
python -m hallulab download
python -m hallulab ingest
```

选择调用方式（已有 `.env` 时直接编辑，避免覆盖本地配置）：

```bash
cp .env.example .env
# 通过编辑器打开 .env，选择调用后端
```

### 使用 ChatGPT 账号额度

已安装 Codex CLI 时，在自己的终端完成浏览器授权：

```bash
codex -c 'forced_login_method="chatgpt"' login --device-auth
```

在 `.env` 中设置：

```dotenv
HALLULAB_BACKEND=codex
CODEX_MODEL=gpt-6-luna
CODEX_REASONING_EFFORT=low
```

然后先验证登录和一次 JSON 生成，再跑 3 道题：

```bash
python -m hallulab doctor --backend codex
python -m hallulab doctor --backend codex --probe
python -m hallulab run --backend codex --limit 3 --out results/pilot_codex.jsonl
python -m hallulab run --backend codex --limit 30 --out results/experiment_codex.jsonl
python -m hallulab review --run results/experiment_codex.jsonl
```

此模式由 Python 调用 `codex exec`，使用账号的 Codex 限额，不会将账号额度兑换为通用 API Key，也不建立 HTTP 转发服务。代码会检查 ChatGPT 登录状态，移除子进程里的 API Key/地址覆盖变量，并强制使用 ChatGPT 认证；不会失败后自动切回 API 计费。登录会使用 CLI 当前的认证存储。

每次请求在空临时目录中建立独立、临时会话，忽略用户 CLI 配置，关闭检索、shell、插件、记忆和其他代理等功能；发现工具事件时拒绝该次结果。实验论文文本只由 B 的检索片段和验证器输入提供。调用审计保存在 `results/codex_calls/*.json`，结果行关联调用 ID，meta 文件记录 CLI 版本、后端和推理设置，禁止与历史 API 结果混用。

`CODEX_MODEL` 必须是该账号实际可用的模型名；当前保留项目原来的 `gpt-6-luna`，尚未通过账号在线验证。若提示不可用，先按账号可用模型修改，不自动替换模型。CLI 带有自身的系统提示，实验报告应写明使用 Codex CLI 后端；它不能视为与裸 Responses API 完全相同的生成条件。当前 `detect` 需要精确控制 `temperature`，仅支持 API 后端；主 A/B/C 实验可使用 Codex 后端。

### 使用 API 余额

在 `.env` 设置 `HALLULAB_BACKEND=api` 和 `OPENAI_API_KEY`，或使用命令行的 `--backend api`。

如果使用的是“ChatGPT 账号转 API”工具或其他网关，需要填写**该工具提供的** Key、Base URL 和模型名。仅填写 Key 时，SDK 默认请求 `https://api.openai.com/v1/`，不会自动找到本地转换服务。请按照工具说明填写路径；当前代码调用 `/responses`，工具必须支持 Responses API。只提供 `/chat/completions` 的接口需要另外适配。

```dotenv
OPENAI_API_KEY=YOUR_GATEWAY_KEY
OPENAI_BASE_URL=http://127.0.0.1:YOUR_PORT/v1
OPENAI_MODEL=YOUR_GATEWAY_MODEL
```

先检查配置，再按需发送一次小型生成请求（可能计费；不运行课程实验）：

```bash
python -m hallulab doctor --backend api
python -m hallulab doctor --backend api --probe
```

诊断输出会隐藏 Key。`401` 表示认证失败；`404` 需要检查地址、模型和接口支持；`429` 要区分 `insufficient_quota` 与临时限流。网络/DNS 错误表示没有收到 API HTTP 响应，不能据此判断余额。余额必须属于当前接口实际使用的平台/项目；ChatGPT 订阅、官方 API 和转换服务的额度要分别核对。

先使用 3 道题验证运行真实 API，再跑完整 30 道题：

```bash
python -m hallulab run --backend api --limit 3 --out results/pilot.jsonl
python -m hallulab run --backend api --limit 30 --out results/experiment.jsonl
```

`run` 默认 BM25，固定 `k=4`。若中途中断，**同样的命令可以断点继续**；若要重新开始，添加 `--restart`。同一个结果文件不能混合不同实验配置。

生成给人工标注的 CSV：

```bash
python -m hallulab review --run results/experiment.jsonl
```

人工在 `results/review/claims_review.csv` 中填写：

- `human_factual_correct`: `1` 或 `0`，依据**完整原始论文/独立可信证据**判断事实是否正确；不要直接照抄验证器标签。
- `human_citation_supported`: 对 B/C 填 `1` 或 `0`，只能基于**该陈述实际标注的引文片段**判断支持关系。A 无引用，不填写。

在 `results/review/questions_review.csv` 中填写：

- `human_answerable_in_corpus`: `1`/`0`，必须检查整个语料后填写，不能根据题目类型推断。
- `gold_claims_total`: 预先人工列出的该问题必要事实数量。
- `gold_claims_covered`: 本组正确回答的必要事实数量（不能超过 total）。

注：这些预填问题的 `gold_answer` 与 `gold_evidence_ids` **故意留空**，因为要依据实际下载的 PDF 版本逐一确认。项目未伪造标注或实验结果。若某论文未成功下载，要补齐后再标 gold。

还需要在 `results/review/citations_review.csv` 对每条引用填写：

- `human_citation_useful`: `1`/`0`，判断这一条引用是否确实支持事实陈述、或在联合引用时提供必要的额外证据；这是简化的引用有用性精确率，**不等于 ALCE 原论文的完整 Citation Precision 实现**。

如希望计算检索指标，还应对有确定 gold evidence 的题目，在 `questions_review.csv` 填写：

- `gold_evidence_ids`: 逐个核对 `data/corpus.jsonl` 中的片段 ID，以空格分隔。程序与该题 `retrieved_evidence_ids` 求交计算 Evidence Recall@k；多个可替代证据片段要单独核验。

所有应填事实和引用标签完成后才计算报告指标：

```bash
python -m hallulab score
# 若只标了部分样本，允许探索性计算；不应当作全样本正式结果：
python -m hallulab score --allow-partial
```

输出 `results/metrics/metrics.csv`, `metrics.json` 与柱状图 PNG。缺失标签不被记作 0，也不把自动验证器的标签当事实金标准。缺失 gold 标注时 `gold_claim_recall` 为 `null`。

## 可选：SelfCheck 风格诊断

```bash
python -m hallulab detect --backend api --run results/experiment.jsonl --limit 12 --samples 5
```

注意：这会额外调用生成模型、语义判断模型多次，因而成本高于主实验。它输出 `results/detection.jsonl`，只是**近似版** SelfCheck 矛盾率和离散语义熵。若要报告 AUROC，必须对检测目标的错误/正确状态另做人工标注，且样本规模过小时应仅作定性探索。

## 可选：混合检索

默认仅 BM25，免下载大规模权重。要使用词汇 + 神经向量混合检索，可运行：

```bash
pip install -r requirements-hybrid.txt
python -m hallulab run --retriever hybrid --out results/experiment_hybrid.jsonl
```

首次加载 `BAAI/bge-small-en-v1.5` 需要从 Hugging Face 下载模型；`hybrid` 与 BM25 属于**不同检索设置**，不应直接混入 A/B/C 主对照。推荐主要报告只固定其中一种。

## 基线、公平性与局限

1. A、B 使用同一生成模型，C 只接收 B 的结果；这样 B→C 的差异来自核验过滤。
2. 当前模型的 JSON `claims` 可能遗漏原回答中的事实，报告应人工抽查 answer 和 claims 是否对齐。
3. 把无充分证据的事实删除可能提高 precision，但会降低 answer coverage 和 gold claim recall；不要只报告 precision。
4. 评估需要独立人工核对。NLI 或 LLM-judge 的 `SUPPORTED` 可能误判；即使引用支持也不保证源文件真实。
5. 从全量论文随机切片的 BM25 可能取到参考文献而不是正文，需做检索误差分析；可在有余力时增加标题/章节感知与 reranker。
6. 原始 `30` 道题是**问题草案**，并非已标注 benchmark。6 道“unanswerable”问题的不可回答性需根据完整语料再次核对。
7. 所有方法均为小规模课堂实验，不应宣称可以推广到开放域所有事实幻觉。

## 输出文件

2026-11-07 已通过原生 Codex 独立会话完成一轮 30 题主实验。见 [运行报告](results/native_codex_20261107_001/REPORT_CN.md) 和 [配对输出](results/native_codex_20261107_001/experiment.jsonl)。该轮采用每批五题的原生会话设置，完整输入、原始输出和会话清单均保存在同目录；独立人工事实标注仍待完成。

- `data/papers_manifest.json`: 8 篇公开论文链接
- `data/questions.jsonl`: 30 个按类别组织的测试问题草案
- `data/corpus.jsonl`: 下载论文后生成的带 `paper_id`/`page` 引用块
- `results/experiment.jsonl`: 每道题的 A/B/C、证据 ID 与验证器标签
- `results/review/*.csv`: 事实、引用、问题三份独立人工评估模板
- `results/metrics/*.csv`, `*.png`: 人工标注后的报告指标

## 源论文

SelfCheckGPT (EMNLP 2023), ALCE (EMNLP 2023), FActScore (EMNLP 2023), Self-RAG (ICLR 2024), Semantic Entropy (Nature 2024), RAGAS (2023), ARES (NAACL 2024), original RAG (NeurIPS 2020). PDF 与出处见 manifest。
