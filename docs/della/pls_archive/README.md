# pls 项目归档（2026-10-01 结题）

结论与全部结果见上一级的 `../pls.md`。这里存放过程材料，checkpoint 已全部删除，指标都在 W&B `reself/marin-della`。

| 路径 | 内容 |
|---|---|
| `BOOTSTRAP.md` | 任务说明：用户要求、仓库背景、研究方法说明 |
| `llal_reading_notes.md` | LLAL 两篇博客、Notion 原文与 arXiv 2504.15471 / 2309.04827 / 2601.07372 的阅读笔记 |
| `survey_intermediate_supervision.md` | 中间层监督论文调研（含审计备注）；`survey_notes/` 是按方法族的分组笔记与核实过的元数据 |
| `audit/` | 用户要求的"严格审计"：报告 `AUDIT.md`，以及审计用的脚本和日志（变异测试的副本 `mut/` 未存，`mutate.py` 可重建） |
| `analysis/` | 分析脚本：`step_compare.py`（逐 eval 步对比 8 个基线 run）、`watch_pls.sh`（作业监控）、`dyn/`（动态指标分析：`fetch.py` 拉逐步指标，`analyze1-3.py` 早期窗口与种子差异，`probe_report.py` / `icl_fine.py` 诊断 run，`story_fig.py` 汇总图；`*.npz` / `*.npy` 是从 W&B 拉下的缓存）、调研与审计用的若干核对脚本 |
| `figs/` | 诊断 run 的梯度余弦与上下文学习分数图（汇总图在 `../pls_figs/130m_dynamic_signals.png`） |
| `gradflow.html` | 给用户讲梯度流的动画页（基线 / 共享头 / 只读共享头） |
| `ledger.yaml` | 作业台账副本（原件 `logs/pls_qwen3_h100x4.yaml` 按惯例不进 git） |
| `logs.tar.gz` | worktree 的 `logs/` 整个打包：每个 run 的 levanter 日志、全部 slurm 作业输出、台账原件 |
| `../pls_runs/<run_id>/` | 每个训练 run 的 `eval_metrics.jsonl`、运行配置 `executor_info.json`、`artifact.json`（checkpoint 删除前留下） |

脚本里的绝对路径指向当时的工作目录 `/scratch/gpfs/GROUP/USER/tmp/pls`（在 della-vis1 上运行）。该目录和 worktree `project/marin-pls` 已在结题时删除；未存入的第三方原文（论文 PDF 与全文提取、博客与 Notion 网页抓取）随之删除，链接见阅读笔记。
