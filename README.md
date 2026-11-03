# SpecRAG

SpecRAG 正在重构为一个以证据为基础的研究智能体，用于研究 3GPP 文档等复杂技术规范。

## 范围

问题和研究任务会直接进入 RAG 系统，不再由已经移除的风险种子、缺陷发现、协议还原或攻击链流水线生成。

目标流程如下：

```text
用户问题或任务
-> Researcher 首轮原生函数调用
-> 执行可信检索工具
-> 有界的观察与调整循环
-> 候选证据
-> 来源与充分性验证
-> 有界缺口检索
-> 以证据为基础的答案
```

## 当前有效代码

- `src/specrag/retrieval/corpus.py`：发现 3GPP 版本，并忠实于来源地切分 Markdown。
- `src/specrag/retrieval/corpus_sync.py`：非破坏式版本发现、以文档为单位的“先构建后激活”状态转换，以及提供商/物化器边界。
- `src/specrag/infrastructure/threegpp.py`：显式发现 3GPP 官方归档、物化 ZIP、安全解压，以及将 DOC/DOCX 转换为 Markdown。
- `src/specrag/retrieval/lexical.py`、`semantic.py` 和 `fusion.py`：BM25、稠密候选搜索和确定性的倒数排名融合。
- `src/specrag/retrieval/hybrid.py`：小型混合检索编排器；语义搜索不可用时，会回退到经过测试的词法检索。
- `src/specrag/retrieval/clauses.py`：绑定 Release、规范和版本的确定性条款索引，包含来源范围、直接及后代分块、父子边和解析后的交叉引用。`clauses.jsonl` 与每个兼容的分块快照一起持久化，且不会改变向量标识。
- `src/specrag/retrieval/planner.py`：稳定的公共门面和高层检索工作流编排。
- `src/specrag/retrieval/routing.py`：为保留的确定性检索工作流执行候选规范路由。
- `src/specrag/retrieval/execution.py` 和 `ranking.py`：候选检索、引用扩展、确定性评分和可选重排。
- `src/specrag/retrieval/relation_evidence.py`、`curation.py` 和 `expansion.py`：有依据的关系发现、确定性整理和有界二阶段检索。
- `src/specrag/retrieval/evidence_selection.py`、`answer_synthesis.py` 和 `selected_answer.py`：来源 ID 选择和证据支持的回答路径。
- `src/specrag/retrieval/*_models.py`：从 Planner 中抽离的不可变规划、执行及答案/证据选择契约。
- `src/specrag/retrieval/hypothesis.py`：推测性搜索指引的稳定门面，由独立的契约、执行和合并策略模块支持。
- `src/specrag/retrieval/reranker.py`：可选的交叉编码器重排。
- `src/specrag/retrieval/normalization.py`：协议标识符规范化。
- `src/specrag/agents/`：独立的 Researcher 模型、证据状态、原生调用策略，以及通过统一工具注册表执行的有界循环。控制器第一轮直接接收 Planner 任务和严格的原生函数定义，并生成初始查询与调用；后续轮次根据已观察到的证据进行调整。检索分块进入去重后的候选池；控制器维护有界的活跃证据集，可以用后续能够弥补缺口的证据替换较弱的早期证据，同时保留完整审计轨迹。每个任务还维护稳定的结构化证据需求，包含缺失、部分覆盖和已覆盖状态，并在正常轮次、检查点及纠正尝试之间保留直接证据链接。每个 `ResearchTask` 拥有初始检索额度，以及一份用于解决 Sufficiency Judge 未处理缺口的独立有界纠正额度；两者都只对实际执行的检索调用计费，并分别记录。系统不设置按工具类别划分的配额。检索预算耗尽后，控制器可以再进行一次仅用于收尾的轮次以检查最后结果，但不能继续调用工具。
- `src/specrag/prompts/researcher.py`：带版本的 Researcher 控制器提示词契约。
- `src/specrag/prompts/synthesis.py`：带版本的最终答案契约。原始问题具有最高权威性，任务问题是覆盖检查点，只有来源有效且被 Sufficiency Judge 选中的证据才具备使用资格。
- `src/specrag/evidence/synthesis.py`：验证每个已覆盖任务都由合格的行内证据引用表示。同一个 Synthesizer 会生成完整答案，或带明确限定的部分答案，并标注已覆盖、未解决和冲突的任务 ID；不存在平行的主张列表或主张拼接步骤。
- `src/specrag/tools/`：与提供商无关的逐工具参数架构、函数描述、注册表，以及九个可信检索适配器。条款查找支持基于游标读取长条款，并可显式包含后代条款；父子导航和经来源确认的交叉引用导航均有界、分页，并限制在 Run 的不可变语料范围内。注册表是定义、验证和执行的唯一来源。HyDE 会为检索嵌入控制器生成的不可信段落；只有返回的来源分块才能成为证据。
- `src/specrag/models/`：独立于框架的研究状态契约。
- `src/specrag/context/`：与提供商无关、感知模型的请求预算，带有有界分页的完整持久工具结果，结构化消息历史摘要、压缩检查点和重建。它不会替代 Evidence、Run 状态、语料绑定或 Memory。
- `src/specrag/infrastructure/conversation_store.py` 和 `conversation_queue.py`：由迁移支持的私有 Conversation 持久化、规范消息/Turn/事件的原子提交，以及按 Conversation 租约管理的 Turn 队列。Context 消息仍是唯一持久的消息历史。
- `src/specrag/infrastructure/result_store.py`：不可变且感知尝试次数的 Research Result 持久化。终止态 Run、对应 Outcome 和一项完成发件箱作业会在 Result 物化前原子提交；Result 回读从不依赖 LangGraph 检查点。
- `src/specrag/infrastructure/conversation_lifecycle_store.py` 和 `completion_queue.py`：由后端拥有的完成协调器边界。持久回执使发件箱确认可以安全重放；原子对账会把 Run/Result 状态链接到有序 Conversation Event，并按条件排入一个自动展示 Result 的 Turn。
- `src/specrag/interfaces/mcp.py`：仅供内部回环访问的 Streamable HTTP MCP 服务器和标准 MCP 客户端封装。三个研究工具会委托给现有 Run Worker、生命周期协调器和 Result 服务；传输层不实现任何检索或证据逻辑。
- `src/specrag/access/`：默认拒绝的工作区授权，以及本地密码和会话服务。密码使用 `scrypt`；数据库只存储会话令牌摘要。本地 CLI 将持有者令牌保存在被忽略的 `.local/state/session-token` 凭据缓存中。
- `src/specrag/specfacts/`：可复用规范事实的待处理提案、审查者批准、冲突、取代、失效和授权提示查找。
- `src/specrag/memory/`：用户核心记忆、工作区项目核心记忆和工作区主题记忆。核心记忆和已固定主题会在有界上下文预算内默认入选；未固定主题使用小型本地 BM25 召回路径。Memory 永远不会变成 Evidence 或 SpecFact。显式写入带版本且幂等；Extractor 输出在所有者/审查者批准前保持待处理；每个 Run 都会记录实际使用的授权修订版本。低频整合必须同时满足 24 小时间隔和 10 项已接受变更；容量维护只会为未固定的主题条目提出可逆归档建议。
- `src/specrag/orchestration/`：LangGraph `StateGraph`、条件路由、依赖上下文交接、有界重新规划、SQLite 检查点、授权 Run 生命周期、事件、取消、截止期限和恢复。
- `docs/architecture/single_research_workflow.mmd` 和 `docs/notebooks/langgraph_workflows.ipynb`：受版本控制的业务流程源文件，以及只编译不执行的 Single/Parallel Researcher 图拓扑检查。生成的图表保存在被忽略的 `artifacts/diagrams/` 下。
- `src/specrag/interfaces/`：产品 CLI、HTTP API 适配器和应用组合。API 验证服务器签发的持有者会话，在每个受保护请求中解析当前工作区成员身份，并复用与 CLI 相同的访问策略和服务。
- `src/specrag/infrastructure/structured_agent.py`：结构化智能体运行时适配器。
- `src/specrag/infrastructure/researcher_controller.py`：通过小型决策接口，将带版本的 Researcher 提示词绑定到结构化智能体适配器。
- `src/specrag/infrastructure/llm.py`：兼容 OpenAI 的结构化客户端，包含脱敏回执、逐模型 Token 预检、重试策略和 Responses API 原生函数调用传输。使用 `store=False` 时，各轮次之间只显式重放函数调用及与之匹配的 `function_call_output` 项。
- `src/specrag/infrastructure/context_store.py`：规范 Context 消息、完整工具结果和原子激活压缩检查点的 SQLite 适配器。压缩后仍保留原始历史。
- `scripts/evaluate_first_turn.py`：不再次调用模型，直接执行冻结的 Researcher 首轮产物，并报告严格的已标注来源召回率，供离线检查。
- `scripts/evaluate_answers.py`：针对整理后的 QA 目标执行确定性的答案/证据对齐，写出逐问题结果以及按难度和推理范围分组的汇总报告。它不调用 LLM；可选的结构化答案质量量表会单独报告。
- `scripts/evaluate_end_to_end.py`：起草、人工审查并汇总完整工作流的二元 Task Success 判定，同时从同一 Run 产物中提取延迟、总计/逐调用 Token 用量、LLM 调用数和有序工具调用详情。缺失的 Token 遥测保持为 `null`；由于失败尝试不提供可靠用量，包含重试的 Run 也会被保守标记为未完成。
- `scripts/evaluate_components.py`：用带版本的标注和结构化语义判定评估冻结的组件产物。支持 `router`、`planner`、`retriever`、`researcher`、`judge` 和 `synthesizer`；提议状态的标注必须显式使用 `--allow-proposed`，且生成的报告会标记为 `calibration_only`。
- `scripts/draft_component_evaluation.py`：使用带版本的结构化离线 Judge 提示词，起草语义单元标注和 Planner 映射，供人工审查。生成的草稿绝不会自行提升为 `reviewed`。
- `scripts/run_component_researcher.py`：直接使用经审查的 GoldResearchTask 运行生产 Researcher，冻结逐问题证据和工具调用产物，不继承 Planner 错误。
- `scripts/run_component_evidence_pipeline.py`：从独立的黄金充分候选池运行 Judge，并从另一份已批准证据输入运行 Synthesizer，从而保持指标隔离。
- `scripts/draft_component_retrieval.py` 和 `scripts/draft_component_evidence.py`：创建带版本的结构化 qrel、证据集、Final Evidence 和主张/引用判定草稿。
- `scripts/promote_component_judgments.py`：记录明确的人工审计决定并添加必要的审查者元数据；未提供 `--audit-complete` 时拒绝提升。

  同一个入口点可以将现有试点合并为完整 Router 标注种子。旧审查输入必须保持 `proposed`；提升为 `reviewed` 需要可审计的审查者和包含时区的时间戳。

  ```powershell
  uv run python scripts/evaluate_components.py seed-router `
    --questions data/QA/questions.jsonl `
    --route-review artifacts/runs/routing/20261010-router-label-review/review.jsonl `
    --base-annotations data/QA/component_evaluation_pilot.jsonl `
    --output data/QA/component_evaluation_release17.jsonl
  ```

  ```powershell
  uv run python scripts/evaluate_components.py router `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --artifacts artifacts/runs/routing/20261010-all-50-v37-contract-cohesion `
    --output artifacts/evaluations/components/router-pilot-reviewed
  ```

  固定组件标注后，无需重新运行 Router 即可为校准集生成 Planner 候选产物：

  ```powershell
  uv run specrag plan --id r17-qa-035 --id r17-qa-044 `
    --route-annotations data/QA/component_evaluation_pilot.jsonl `
    --output-dir artifacts/runs/planning/component-pilot
  ```

  评估五个试点分解用例时，将该目录与冻结的 Planner 批次组合使用：

  ```powershell
  uv run python scripts/evaluate_components.py planner `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --judgments data/QA/component_planner_pilot_judgments.jsonl `
    --artifacts artifacts/runs/planning/20261010-complex-planner-v13-full-acceptance `
    --artifacts artifacts/runs/planning/component-pilot `
    --output artifacts/evaluations/components/planner-pilot-reviewed
  ```

  经审查的完整开发集输入为 `data/QA/component_evaluation_release17.jsonl` 和 `data/QA/component_planner_release17_judgments.jsonl`。保留的 Router 与 Planner 报告分别覆盖全部 50 个路由用例和 18 个分解用例，均为 `calibration_only=false` 的正式运行。

  经完整审查的证据路径输入为四个 `data/QA/component_*_release17_judgments.jsonl` 文件。正式基线覆盖 78 次 Retriever 调用、78 个 Researcher 任务，以及 50 个 Judge/Synthesizer 问题。Retriever Recall@12 为 0.8761、nDCG@5 为 0.8738；Researcher Coverage/Precision 为 0.9872/0.6178；Judge Coverage/Precision/Overlap 为 0.98/0.908/0.102；Synthesizer Coverage/Groundedness/Citation Correctness 为 0.975/0.9937/0.9567。所有报告都具有完整的已审查计数，且 `calibration_only=false`。

  证据路径组件共用同一入口点和冻结的运行时产物。每条命令接收对应组件的结构化判定文件；Retriever 还会拒绝导航工具调用，并记录固定语料范围。

  ```powershell
  uv run python scripts/evaluate_components.py retriever `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --judgments data/QA/component_retrieval_pilot_judgments.jsonl `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-simple-34 `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-complex-16 `
    --output artifacts/evaluations/components/retriever-pilot-reviewed

  uv run python scripts/evaluate_components.py researcher `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --judgments data/QA/component_researcher_pilot_judgments.jsonl `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-simple-34 `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-complex-16 `
    --output artifacts/evaluations/components/researcher-pilot-reviewed

  uv run python scripts/evaluate_components.py judge `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --judgments data/QA/component_final_evidence_pilot_judgments.jsonl `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-simple-34 `
    --artifacts artifacts/runs/researcher-freeze/20261013-native-complex-16 `
    --output artifacts/evaluations/components/final-evidence-pilot-reviewed

  uv run python scripts/evaluate_components.py synthesizer `
    --annotations data/QA/component_evaluation_pilot.jsonl `
    --judgments data/QA/component_synthesizer_pilot_judgments.jsonl `
    --artifacts artifacts/runs/synthesis-v12/20261014-simple `
    --artifacts artifacts/runs/synthesis-v12/20261014-complex `
    --output artifacts/evaluations/components/synthesizer-pilot-reviewed
  ```

- `scripts/evaluate_complex_researcher.py`：运行单个冻结的复杂 Planner 产物，或运行可恢复的目录批次。批处理模式会原子保存每个问题和 `summary.json`；中断后使用 `--batch --resume` 重新运行，可跳过有效的已有产物并继续未完成工作。

  ```powershell
  python scripts/evaluate_complex_researcher.py `
    artifacts/runs/planning/<accepted-planner-run> `
    artifacts/runs/complex-freeze/<batch-run> `
    --batch --resume
  ```

历史攻击者能力验证线路及其带版本智能体、确定性门禁、Gold/Shadow 评估、提示词、用例和基准测试均已移除。当前 Retriever 只索引 3GPP Markdown 语料；已删除的威胁模型不再作为输入来源。

## 组件评估基线

经审查的内部基线于 2026-10-17 在包含 50 个问题的 Release 17 开发集上完成。它是工程回归基线，并非独立学术基准。所有正式报告都使用经审查的标注与判定，并设有 `calibration_only=false`。

| 组件 | 范围 | 经审查的基线 |
| --- | ---: | --- |
| Router | 50 个问题 | Route Accuracy 0.9000；Macro-F1 0.8927 |
| Planner | 18 个分解问题 | Task Coverage 0.8241；Full Task Coverage 0.6111；Overlap 0；Out-of-Scope 0 |
| Retriever | 78 次固定调用 | Target Recall@5/@12 0.8761；nDCG@5/@12 0.8738 |
| Researcher | 78 个 GoldResearchTask | Evidence Coverage 0.9872；Evidence Precision 0.6178 |
| Judge | 50 个黄金充分候选池 | Final Evidence Coverage 0.9800；Precision 0.9080；Overlap 0.1020 |
| Synthesizer | 50 份独立于 Judge 的证据输入 | Answer Coverage 0.9750；Groundedness 0.9937；Citation Correctness 0.9567 |

`Full Task Coverage` 有意采用严格定义：只有当每个经审查的 ResearchUnit 都至少被一个生成的 ResearchTask 覆盖时，该问题才算通过。Planner 在 18 个分解问题中的 11 个覆盖了全部单元，其逐问题平均 Task Coverage 为 0.8241。

各组件相互隔离，因此后续阶段分数不是端到端分数：Planner 接收经审查的路由；Researcher 接收固定的 GoldResearchTask；Judge 接收黄金充分候选；Synthesizer 接收独立于运行时 Judge 选择的已批准证据。当前 Retriever 产物为每次固定调用提供五个排序结果，所以报告中的 `@12` 与 `@5` 相等；未来比较必须先检索十二个结果，才能解释额外深度。

指标契约和审查决定记录在 [`docs/plans/031-component-evaluation-system.md`](docs/plans/031-component-evaluation-system.md) 与 [`docs/component_evaluation_review.md`](docs/component_evaluation_review.md)。经审查的标注和判定位于 [`data/QA/`](data/QA/README.md)。生成的运行和报告保存在被忽略的 `artifacts/` 下；已签入的输入和 `scripts/evaluate_components.py` 可以复现报告。未来的优化提交应记录发生变化的提示词、模型或语料契约，并同时报告新分数及其相对本基线的变化量。

## 端到端与效率评估

当前 Final-answer Benchmark V1 将同一批 50 个问题与经审查的 Reference Answer 和 Answer Unit、自审后的加权 Rubric、已知充分但非排他的 Gold Evidence，以及精确的 34 文档 Release 17 Environment 组合在一起。它只区分 `simple`（34 个）和 `complex`（16 个），每个用例冻结一次试验，并公开防泄漏的 Workflow 输入，其中仅包含任务 ID、Question 和 Environment。使用以下命令验证：

```powershell
uv run python scripts/check_end_to_end_benchmark.py
```

Benchmark 契约和整理规则记录在 [`docs/end_to_end_benchmark.md`](docs/end_to_end_benchmark.md) 和 [`docs/plans/044-final-answer-benchmark.md`](docs/plans/044-final-answer-benchmark.md)。完整的 50 用例人工审查视图以单个已签入文件形式保存在 [`data/QA/end_to_end_benchmark_review.json`](data/QA/end_to_end_benchmark_review.json)；标准检查会保证它与组合后的源数据一致。

Final-answer Judge 已将提示词冻结为 `final-answer-rubric-judge/v1.1`，位置为 [`src/specrag/prompts/rubric_judge.py`](src/specrag/prompts/rubric_judge.py)。它只评判 Question、Reference Answer、Rubric 和 Candidate Answer，并返回有序条目等级，而不是由模型生成的分数。严格模型会拒绝缺失、重复或乱序的 Rubric ID。应用代码附加各项 Benchmark 权重，将 `full/partial/none` 映射为 `1.0/0.5/0.0`，并计算 0–100 分。已保存答案的批处理运行器已经实现；人工校准仍待完成。参见 [`docs/plans/045-final-answer-judge.md`](docs/plans/045-final-answer-judge.md)。

有界执行策略也已实现。空白 Workflow 答案会绕过模型，以确定性方式得到零分，且每项都标为 `none`。非空答案最多允许针对同一不可变输入进行两次顶层 Judge 尝试。如果两次尝试都验证失败或提供商失败，执行会记录 `evaluation_failed` 和空分数，而不是错误报告零分；无需重新运行 Research Workflow，即可再次评判已保存的答案。人工校准仍待完成。

现在可以通过 `scripts/evaluate_final_answer_benchmark.py` 评估已保存的 Workflow 产物，而无需重新运行研究。运行器支持 `simple`、`complex` 或全部冻结用例、可恢复输出、API 或隔离 Codex 后端、源产物哈希冻结、逐用例执行记录，以及总体和按难度分组的分数汇总。生成的判定保留在被忽略的 `artifacts/` 树下。

一次试验的模型比较入口也已实现。当前候选模型固定为 `gpt-5.6-sol`，全部模型驱动节点使用该候选模型，跨模型 fallback 被禁用，最终答案统一由 `deepseek-v4-pro` 盲评。对 429、超时和暂时性 Provider 故障最多重试三次，并采用 5/10/20 秒指数退避；协议错误不会重试。本轮只汇总 Judge 分数、覆盖率和运行统计，不计算 Task Success、Pass@k、Pass^k 或阈值通过率。LangSmith 用于节点和模型调用轨迹监督，本地结构化 Run、Evidence、回执与 Judge 产物仍是评测权威数据。

```powershell
uv run python scripts/evaluate_model_comparison.py probe artifacts/evaluations/model-comparison/EXP-ID
uv run python scripts/evaluate_model_comparison.py run artifacts/evaluations/model-comparison/EXP-ID --resume
uv run python scripts/evaluate_model_comparison.py judge artifacts/evaluations/model-comparison/EXP-ID --resume
uv run python scripts/evaluate_model_comparison.py summarize artifacts/evaluations/model-comparison/EXP-ID
```

正式探针前必须配置并轮换 `.env` 中的 OpenAI-compatible 代理和 LangSmith 凭据。能力探针验证严格结构化输出和 Researcher 两轮原生工具 replay；必选模型失败会阻止正式批次，可选 Claude 失败则明确记录并跳过。完整契约见 [`docs/plans/046-multi-model-langsmith-evaluation.md`](docs/plans/046-multi-model-langsmith-evaluation.md)。

端到端基线使用同一批 50 个可回答的 Release 17 问题，但不会注入经审查的中间任务或证据。其唯一质量汇总指标是 `Task Success Rate`：Run 必须生成正常答案，在没有重大错误或无依据结论的情况下回答核心结论，正确引用支持核心结论的证据，并遵守请求的语料范围。每项决定都经过人工审查；失败和预算耗尽的 Run 仍计入分母。

执行先经过十问题试点 `001`、`002`、`005`、`019`、`034`、`007`、`008`、`035`、`044` 和 `046`，然后覆盖全部 50 个问题。基线命令会记录问题集哈希、Git 修订版本、预算、精确语料文档及哈希、索引指纹、提示词/模型回执和热缓存条件。每个逐问题 Run 也携带完全相同的文档快照；Task Success 会拒绝路径、版本、Release 或文档哈希不在快照内的引用。评估命令如下：

```powershell
uv run python scripts/evaluate_end_to_end.py draft RUN_DIR `
  data/QA/questions.jsonl data/QA/component_evaluation_release17.jsonl EVAL_DIR
uv run python scripts/evaluate_end_to_end.py review EVAL_DIR REVIEW_DECISIONS.jsonl
uv run python scripts/evaluate_end_to_end.py summarize EVAL_DIR
```

契约和当前执行状态记录在 [`docs/plans/032-end-to-end-efficiency-evaluation.md`](docs/plans/032-end-to-end-efficiency-evaluation.md)。已完成且经人工审查的正式基线为 **38/50 = 76.0% Task Success**。平均/P95 延迟为 79.183/191.969 秒，共有 450 次 LLM 调用和 289 次工具调用尝试。50 个 Run 中有 49 个具备精确 Token 遥测；另一个发生重试的逻辑调用缺少四次失败提供商尝试的用量，因此精确批次 Token 总量有意保留为 `null`，避免低报。效率表、全部失败用例、审计修正、限制和复现记录参见 [`docs/end_to_end_evaluation.md`](docs/end_to_end_evaluation.md)。

## 数据

- `data/3gpp/`：保留的全部 3GPP 规范版本和源格式。
- `data/3gpp/manifest.json`：紧凑语料清单。
- `data/3gpp/catalog.jsonl`：每个规范版本一条目录记录。
- `.local/state/specrag.sqlite3`：被忽略的本地账户、工作区、成员关系、会话、Run、事件、检查点、SpecFact、核心/主题 Memory、提案、语料版本记录、活动指针、Run 语料绑定和维护游标。SQLite 启用外键和 WAL，并应用编号架构迁移。
- `data/QA/questions.jsonl`：整理后的 Release 17 开发问题及精确本地证据绑定；完整上游快照保存在 `data/QA/` 下被忽略的同级目录中。

生成的索引写入 `.local/indexes/specifications/`，并被 Git 忽略。它们是可复现缓存，而不是项目数据。

## 配置

将 `.env.example` 复制为 `.env`，设置兼容 OpenAI 的端点和模型 ID，然后在 `config.yaml` 中调整检索机制。

当前产品研究范围固定为 Release 17。接受研究任务前，先初始化其版本注册表，并对已经构建的分块/向量产物执行一次验证：

```text
specrag corpus sync --release 17 --apply
specrag corpus status --release 17
```

之后，新 Run 会冻结当前活动的 Release 17 版本；恢复时即使活动指针已经变化，也会复用完全相同的绑定。如果 Release 17 未激活或其只读检索快照不完整，Run 会被明确拒绝；研究执行过程绝不会构建分块或添加嵌入。其他 Release 和跨版本研究属于后续产品能力。

规范同步默认有意保持休眠。本地 CLI 会扫描 `data/3gpp` 下已经物化的 Markdown；显式使用 `--official` 时，才通过 3GPP 官方 FTP 归档适配器发现当前修订版本。除非提供 `--apply`，否则 `corpus sync` 只是只读预览。应用操作只会下载并物化需要活动版本的候选项，然后只为每个 Release/规范构建最新候选文档，验证分块、BM25 和向量完整性，再切换活动指针。构建失败时保留旧活动版本。切换成功后，旧版本标记为 `draining`；源文件和索引都不会自动删除。稠密向量保存在以嵌入/分块配置为键的追加式缓存中；摄取只嵌入缺失的分块 ID，而每次查询都根据对应 Run 快照中的精确文档哈希过滤这份共享缓存。

产品 CLI 包含以下本地多用户命令：

```text
specrag auth bootstrap --username alice
specrag auth login --username alice
specrag workspace create --name standards
specrag workspace member add --workspace standards --username reviewer --role reviewer
specrag research run --workspace standards --question "..." [--release 17]
specrag research inspect|resume|cancel --workspace standards RUN-...
specrag specfact review --workspace standards SFP-... --approve
specrag memory set|list|delete --workspace standards ...
specrag memory topic create|list|add|pin|unpin --workspace standards ...
specrag memory recall --workspace standards --query "reranker experiment"
specrag memory extract --workspace standards --source-ref TURN-... --user-text "..."
specrag memory proposal list|review --workspace standards ...
specrag memory maintenance status|consolidate|prune|restore --workspace standards ...
specrag corpus status [--release 17]
specrag corpus sync [--release 17] [--apply] [--official]
```

首个 Web/API 安全边界也已经可用。先通过 CLI 引导创建第一位本地管理员，再让 API 监听安全的仅本地默认地址：

```text
specrag auth bootstrap --username alice
specrag-api
```

`POST /api/v1/auth/login` 返回服务器签发的持有者令牌。受保护请求通过 `Authorization: Bearer <token>` 传递令牌；API 从不接受客户端提供的可信用户 ID、角色或会话 ID。当前版本在 `/api/v1` 下提供登录/登出、工作区列出/创建和上下文、成员管理、仅所有者可见的 Conversation CRUD/消息/事件恢复、授权 Run 检查/取消、Memory/SpecFact 仪表板和提案审查。管理员还可以列出工作区目录、添加成员、更改其他成员角色和移除其他成员关系；服务拒绝用户降低或移除自己的权限。User Core 响应仅所有者可见；工作区知识限定在工作区范围，审查操作由现有角色策略强制执行。

现在也可以通过 `POST /api/v1/workspaces/{workspace}/runs` 提交研究；API 返回 HTTP 202，以及排入现有 RunHarness 的 Run。客户端可通过 `GET /api/v1/workspaces/{workspace}/runs/{run_id}/events` 以 SSE 消费有序持久事件。API 还提供通过身份验证的取消，以及异步 Resume 和 Retry 控制。Resume 需要持久检查点并继续同一个 Run；Retry 会创建新 Run，并保存其 `retry_of_run_id` 关系。检查只返回脱敏后的失败摘要，包括安全的失败类型、是否适合手动重试，以及生命周期指标和持久事件，而不会返回异常堆栈或提供商载荷。API 与 Worker 线程使用同一 WAL 数据库上的独立 SQLite 连接，而不是并发共享一个连接。系统有意不启用凭据 Cookie 或宽松 CORS；绑定到 localhost 以外地址的部署必须终止 TLS。Run 作业保存在带原子租约和启动恢复能力的持久 SQLite 队列中。

在常规 API + MCP 部署中，`specrag-api` 只启动 Conversation Worker，`specrag-mcp` 则拥有 Research 和 Lifecycle Worker。这可以防止两个进程执行或尝试修复同一个研究索引。显式选项 `--worker-mode embedded` 仍可用于隔离测试和单进程诊断。Redis 等代理属于后续横向扩展选项。

持久 Conversation 位于 `/api/v1/workspaces/{workspace}/conversations` 下。创建用户消息会返回 HTTP 202 和已排队的 Conversation Turn。消息正文与 Turn、持久 Turn 作业、聚合水位及有序 Conversation Event 在同一事务中写入现有 `context_messages` 表。消息提交需要操作键并支持预期修订版本；完全相同的重放返回原回执，过期或冲突输入返回 `conversation_conflict`。Conversation 事件端点同时支持 `since` 和 `Last-Event-ID`，用于 SSE 补取。Conversation 及其消息正文仅创建者可见；仅拥有工作区管理员身份并不会获得读取权限。同一规则也适用于 Run 问题、Event、检查点、Result 和证据。运维人员使用不含内容的 `/run-metadata` 和 `/runs/{run_id}/metadata` 投影，可以在不接收私有研究内容的情况下取消 Run。

每个新提交的 Research Run 都会冻结其语义问题、目标、条件、请求的规范范围、历史 Result 引用和 Conversation 来源。执行前会绑定精确语料版本。正常预算或证据不足停止仍保留 `completed` 执行状态，并发布限定于该次尝试的 Outcome；失败、取消和超时仍属于执行失败。后端可将 Outcome 确定性物化为不可变 Research Result，其证据状态为充分、部分、不充分或冲突。Result 仅所有者可见，可通过 `/api/v1/workspaces/{workspace}/research-results/{result_id}` 和对应的 Conversation Result 集合访问。Run 检查会公开最新 `result_id`。

新的 Research Run 默认使用“优先完成”模式：不存在覆盖整个 Run 的墙钟截止时间，但迭代/工具/重规划预算、单次提供商超时、取消、Worker 租约和崩溃恢复仍然有界。运维人员需要时，可以显式设置 `--elapsed-seconds` 来启用兼容的分阶段截止策略。依赖已就绪的独立 ResearchTask 会作为持久 TaskExecution 并发运行。只有当所有规划任务都完成、受阻或失败后才会进入 Synthesizer；正常完成的 Run 仍可能发布部分、冲突或不充分的 Result。MCP 同步等待仍为 10 秒；窗口到期后会把同一个 Run 交给 Worker。

Lifecycle Worker 在嵌入式和独立 Worker 模式下消费完成发件箱。来源 Turn 活跃时会延迟展示；出现更新的用户消息或授权失效时会跳过展示；同时防止两个并发 Result 创建相互竞争的自动 Turn。Conversation 详情响应包含当前 Turn、已审计工具调用和研究链接投影，使重新连接的客户端能把数据库快照与 SSE 补取结合起来。有界 Conversation Agent 和带租约的 Turn Worker 现在会组装 Context Harness 历史、最近确认的研究范围、建议性 Memory 和已授权 Result。每个已提交研究请求都会记录不可变范围修订版本；条件变化会创建新 Run，并保留之前 Result 的哈希关系。普通对话无需 Run 即可回答；基于规范的判断使用 MCP 研究工具或适用的已存储 Result。助手的来源归属和引用会在发布前以事务方式验证。

内部 MCP 门面以 `specrag-mcp` 提供。通过 `SPECRAG_MCP_DELEGATION_SECRET` 在签发应用和 MCP 进程中配置同一个高熵服务密钥，然后使用默认回环地址启动：

```text
specrag-mcp --database .local/state/specrag.sqlite3
```

在完整本地服务会话期间，`specrag-mcp` 是单个长时间运行的进程；不得为每个 Research Run 启动一次。它会在验证活动的 Release 17 注册表，并预热不可变 Retriever、向量索引和嵌入模型的同时发布 HTTP。`GET /healthz` 报告进程状态；只有研究 Worker 就绪后，`GET /readyz` 才返回 200。在 `starting` 或 `warming` 期间，仍可回读 Result/状态，但 `start_research` 会返回稳定的不可用响应。预热失败会报告为 `failed`，而不暴露异常文本。就绪后的 Run 提交只执行由数据库支持的轻量版本绑定；绝不切分或嵌入语料。

它在 `http://127.0.0.1:8766/mcp` 暴露 Streamable HTTP 端点，并为本地部署启动现有 Run 和 Lifecycle Worker。首个版本会拒绝非回环绑定地址。`start_research`、`get_research_status` 和 `get_research_result` 工具永远不接受模型提供的身份、Workspace 或角色字段。后端会签发包含已授权 Conversation 来源和幂等键的短期签名委托；MCP 服务器验证它，并在调用应用服务前重新检查当前成员身份。`start_research` 等待同一个持久 Run 的时间不超过十秒，绝不会把等待超时转换为第二次提交或更短的研究预算。MCP 调用具有有界客户端超时；研究输入限制文本/列表大小；工具参数在写入其私有审计记录前会受到严格验证。Conversation 作业按 Conversation 串行执行，默认每位工作区用户最多同时存在两个活跃 Turn。归档是受支持的保留操作；自动硬删除仍被有意推迟。

`specrag.evaluation.conversation` 为直接回答与研究路由、不受支持的规范性回答、条件保留、Result 选择、引用忠实性、上下文延续和压缩恢复提供确定性 Conversation 轨迹检查。这些检查是现有答案/证据基准的补充；它们不会仅凭工具调用就认定证据有效。

HTTP 失败保留标准 `detail` 字段，并包含稳定的机器可读 `code`；请求形状失败使用 `validation_error`。受支持范围有意冻结为本地 3GPP 工作区流程。外部代理、私有文档上传、额外语料提供商、生产部署和整个 Run 的自动重放均被推迟。

首个浏览器客户端位于 `frontend/`。它是本地 React + TypeScript + Vite 应用：浏览器只在 `sessionStorage` 中保存服务器签发的会话令牌，通过 API 获取工作区成员身份和上下文，绝不直接打开 SQLite 状态。界面默认使用简体中文（`zh-CN`），包括本地化生命周期标签和日期；从浏览器直接提交的 Run 会请求中文研究答案。主研究台由持久 Conversation 列表和线程组成，支持普通对话、多轮规范研究、行内 Turn/工具/Run/Result 状态、来源归属、归档/恢复，以及基于游标的 SSE 重连恢复。旧版直接 Run 入口和关联 Run/证据视图仍然可用。系统还提供响应式 Memory/SpecFact 管理界面。仅管理员可见的“人员与权限”面板可以添加本地成员、编辑角色、确认移除并解释各角色；非管理员客户端不会请求成员目录。

先执行一次 Release 17 语料激活，然后在三个独立终端中持续运行 MCP、API 和前端。MCP 在进程生命周期内拥有全部 Run 共用的 Retriever 池和两个研究 Worker；关闭 MCP 会使新的规范研究暂时不可用，但不会删除已持久化的 Conversation 或 Result。

```text
specrag corpus sync --release 17 --apply

# 终端 1：长时间运行的研究服务
$env:SPECRAG_MCP_DELEGATION_SECRET="<至少32字节的共享密钥>"
specrag-mcp

# 终端 2：长时间运行的应用 API；使用完全相同的密钥
$env:SPECRAG_MCP_DELEGATION_SECRET="<同一个共享密钥>"
specrag-api

# 终端 3：浏览器开发服务器
cd frontend
npm install
npm run dev
```

API 启动时，Conversation 运行时会通过真实 Turn 所使用的同一个已配置模型、Responses API 路径和严格函数架构发送一次有界能力探测。API 的 `GET /healthz` 同时报告脱敏后的 `conversation_agent` 状态（`ready`、`unavailable` 或 `disabled`），以及 MCP `research_service` 就绪状态（`starting`、`warming`、`ready`、`failed` 或 `unavailable`）和逐阶段预热耗时。浏览器会在提交前显示这些诊断；失败 Turn 会保留安全且可操作的原因，而不只显示 `invalid_request`。能力失败时，API 仍以降级模式提供登录、历史和检查功能，同时阻止 Turn Worker 重复已知不兼容的模型请求。此诊断绝不会返回提供商响应正文、凭据、端点、提示词或请求头。

普通介绍可以说明 SpecRAG 用于研究 3GPP 规范和比较 Release，而不会被误判为缺少依据的规范性回答。实际要求性语言在没有适用 Research Result 支持时仍会被拒绝；已知响应契约拒绝现在会生成明确且安全的 Turn 诊断。

进程启动时，模型适配器会读取当前仓库的 `.env`；该文件中非空的 OpenAI 兼容设置优先于继承而来的过期进程值。更改模型密钥、端点或模型后，请重启 `specrag-api`，确保启动探测使用新配置。MCP 委托密钥是独立服务凭据，在 API 和 MCP 进程环境中仍必须完全一致。

不要在常规 MCP 部署旁启动 `specrag-worker`：MCP 已经拥有 Research 和 Lifecycle Worker。`specrag-api --worker-mode external` 与 `specrag-worker` 仅作为另一种独立 Worker 拓扑保留，所有参与进程必须指向同一个 SQLite 状态数据库。

Vite 默认将 `/api` 代理到 `http://127.0.0.1:8000`。API 托管在其他位置时，请设置 `VITE_API_BASE_URL`。完成的 Run 现在会显示带行内证据 ID 的来源支持答案，以及包含丰富来源信息的证据卡片。工作区 Run 列表在刷新后从 SQLite 加载，保留所有者/管理员可见性规则，将活跃 Run 重新连接到 SSE，并允许用户重新打开历史结果。知识视图显示用户/项目/主题 Memory、提案、经审查的 SpecFact、来源定位信息和提案状态。所有者可以审查自己的私有 Memory 提案；审查者/管理员可以批准或拒绝工作区 Memory 与 SpecFact 提案，拒绝时必须填写理由。

打开历史 Run 会读取其独立持久化的 Result，并显示整体证据状态，以及每个 Research Task 的问题、Judge 状态、已选证据数量和未解决关系。界面还会显示检查点可用性、Token/阶段指标、最近事件、安全失败诊断，以及感知状态的 Cancel、Resume 和 Retry 操作。

所有产品研究命令都要求 `--workspace`。受信任会话和成员身份会在访问语料或检查点前生成 `ActorContext`。成员管理自己的 Run 和 User Core。工作区成员读取 Project/Topic Memory 并提交候选项；审查者和管理员发布 Project/Topic Memory、管理固定状态、批准 Memory 提案并进行维护。审查者还会批准工作区 SpecFact；管理员管理成员和其工作区中的全部 Run。缺失上下文或成员身份时拒绝访问。Run ID 与检查点线程 ID 相互独立；线程 ID 永远不会授予访问权限。

现有诊断 CLI 保留 `route`、`plan`、`research-plan` 和 `baseline`。`specrag route` 选择 `single_task/simple` 或 `decompose/complex`，并记录一条简洁理由；`specrag plan` 使用经审查的开发标签隔离并检查有界任务分解，无需重新运行 Router；`specrag research-plan` 生成并验证每个依赖已就绪任务的首批工具调用，但不构建索引或执行检索；`specrag baseline` 则继续执行检索和证据流程。Release 是 Researcher/baseline 的检索参数，不是 Router/Planner 输入。所有命令都可通过 `--id` 选择单个问题，或通过 `--all` 选择完整的 50 问题开发集。LangGraph 仅限于 `specrag/orchestration/`；领域、检索、证据、SpecFact、Memory 和评估逻辑均独立于编排运行时。

研究 Run 使用不透明 ID、持久 SQLite 检查点、显式经过时间和取消停止条件、幂等事件，以及逐阶段延迟/Token 汇总。SpecFact 提取与最终答案综合相互独立：独立蒸馏器可以创建绑定来源的待处理提案，且只有审查者可以批准。完成的 Run 不再自动把综合输出转为提案。已批准事实作为查找来源的提示提供给 Researcher；工作流仍会检索、验证并引用原始规范文本。

用户配置优先级依次为命令参数、活动的私有 User Core、`config.yaml`。Run 会保存启动时使用的精确 SpecFact 和 Memory 修订 ID，所以之后的修订不会改变其研究上下文。

每个规划产物都会记录计划是以确定性方式还是由模型生成、当前 Planner 契约与提示词版本，以及模型生成计划所对应的结构化智能体回执。确定性的简单计划不进行 Planner 模型调用，因此记录空提示词回执。

Planner 依赖表示经过验证的信息流。只有当前置任务证据充分后，依赖的 Researcher 才会启动，并且只接收前置任务选中的、来源有效且具备支持的证据作为限定范围的研究上下文。进入新就绪的依赖波次属于正常执行，不消耗 Gap Re-planner 预算；只有已经尝试但未完成的任务才会消耗该预算。

受支持的证据循环是 `Researcher -> source_validator -> sufficiency_judge`。Researcher 使用 `submit_for_validation` 交出当前活跃证据；这不是完整性决定。完整候选池仍附在 Researcher 执行记录上供审计。`source_validator` 是确定性的，它在不调用模型的情况下检查去重、允许路径、文档哈希、行定位信息和精确摘录。单一任务级 Sufficiency Judge 选择哪些来源有效段落能够支持任务，判断完整性或冲突，并在需要进一步研究时生成具体的 `ResearchGap`。未解决缺口会连同有界的任务局部纠正额度返回 Researcher。控制器会在同一模型轮次中将缺口改写为新的检索调用；不会添加独立的查询改写节点。初始和纠正用量都会保留在运行时产物中。

## 本地环境

项目使用 Python 3.12 和 `uv`。使用以下命令创建或复现仓库本地环境：

```text
uv sync --extra dev --frozen
```

`uv` 会把运行时和开发依赖安装到被 Git 忽略的 `.venv/` 中。通过 `uv run ...` 运行项目命令；`pyproject.toml` 和 `uv.lock` 是权威依赖定义。

## 开发约束

所有生成的变更都必须遵守 `AGENTS.md` 和已记录的包边界。完成变更前运行：

```text
uv run python scripts/check.py
```

质量门禁先验证整理后的 R17 问题集，再运行结构和测试检查。结构检查会拒绝遗留目录、带版本号的模块名称、违反依赖方向的代码，以及超过正常大小上限的 Python 模块。

完整的本地 3GPP 语料和可变 Memory 被有意排除在 Git 之外。紧凑语料清单和目录仍受版本控制，使语料内容可以在不提交数百兆字节源文档的情况下接受审计。
