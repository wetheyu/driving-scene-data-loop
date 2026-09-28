# Driving Scene Mining and High-Value Data Feedback Loop

[![checks](https://github.com/wetheyu/driving-scene-data-loop/actions/workflows/checks.yml/badge.svg)](https://github.com/wetheyu/driving-scene-data-loop/actions/workflows/checks.yml)

An offline, one-person research project that closes an autonomous-driving data
loop on nuScenes and measures every link of it: mine rare temporal scenes with
a formal matcher, select new data under a fixed annotation budget without
seeing labels, retrain, and check on held-out logs whether the selection
genuinely helped. Negative results are kept. [中文简介在文末](#中文简介).

> Under the same annotation budget, does mined data improve a held-out metric
> more than random data — and when it does not, why exactly not?

## Results

| Protocol | Question | Answer |
| --- | --- | --- |
| v0.8 | Does bad-case-similarity selection transfer? | **No.** A Development effect of `+0.0247` (3.7σ) died on held-out logs (`+0.0033`, 0.3σ). The cause was diagnosed, not excused: the mining queries came from the partition that judged them. [Findings §8–13](docs/FINDINGS.md) |
| v0.10 | Does the redesigned loop beat random? | **Yes.** Ensemble-disagreement selection beats the random distribution by `+0.0619 ± 0.0082` corridor AP (**7.6σ**) on ten held-out logs it never touched, under a criterion frozen before any label was revealed; 14% of the extra labels buy 85% of the full-data gain (random: 55%). The yield-maximising selector fails the same bar. [Findings §14](docs/FINDINGS.md) |
| v0.11 | Can a remote VLM replace the metric labeling Oracle? | **At this operating point, yes.** Labels at Macro-F1 `0.494` retrained to 86% of the Oracle's gain, statistically indistinguishable (`−0.018 ± 0.020`), at `$0.026` per window — and label F1 predicted none of that. [Findings §15](docs/FINDINGS.md) |

What these claims cover, and nothing more:

- one dataset (nuScenes), one representation (frozen DINOv2), one small
  downstream model (GRU);
- the v0.10 effect was measured at a deliberately data-scarce operating point
  with a steep learning curve — at the data-rich point the same question
  measured null (the v0.8 row);
- the v0.11 downstream contrast is Development-level, because both held-out
  sets were already spent, and it holds on corridor but not on Macro-AP once the
  student is widened to hidden 128 ([Findings §16, D5](docs/FINDINGS.md)).

The scenario specification is frozen at `v0.8-three-class-loop`; v0.10 and
v0.11 build on it without editing it. The full narrative, in the order the
evidence arrived, is [Findings](docs/FINDINGS.md).

## The Loop

```text
nuScenes v1.0-trainval  (850 scenes, 68 logs, 34,149 CAM_FRONT keyframes)
  -> whole-log split into design / train / development / pool / held-out
  -> visual-eligibility cleaning        (859,857 -> 120,035 annotations)
  -> Strem mines three temporal scenarios -> 30,749 five-frame windows
  -> frozen DINOv2 features -> GRU classifier      (per-class AP, Macro-AP)
  -> selectors rank a label-free public pool under a fixed budget
  -> the Oracle reveals only the selected IDs -> retrain from scratch
  -> seed-paired scoring on a held-out set that is opened exactly once
```

## Scenarios

Three executable specifications, frozen in [`specs/task_spec_v2`](specs/task_spec_v2).
Positive rates are 3.5–5.4% per class: a roughly 20:1 class imbalance, not a
long-tailed distribution (three near-equal minority classes form no tail).
They are frequent urban interactions on purpose — rarer definitions lacked
statistical support at this dataset's size (Gate A rejected them), and the
method itself is agnostic to scenario rarity.

| Label | Formal meaning | Not claimed |
| --- | --- | --- |
| `pedestrian_ego_near_zone_entry` | the same eligible pedestrian moves from ≥ 20 m to ≤ 15 m from ego within 2 s | collision risk, TTC, intent |
| `vehicle_relative_corridor_entry` | the same eligible motor vehicle moves from `abs(y) ≥ 3 m` to `abs(y) ≤ 1.5 m` while `0 ≤ x ≤ 30 m`, within 2 s | an intentional lane change |
| `pedestrian_vehicle_center_proximity_hold` | the same pedestrian and vehicle annotation centers stay within 5 m for ≥ 1 s | body clearance or conflict |

## Data Contract

Whole `log_token` groups are split before any window exists, so no log ever
crosses a partition:

| Partition | Logs | Use |
| --- | ---: | --- |
| Design | 5 | define and audit labels |
| Initial training | 13 | initial training |
| Development | 7 | early stopping, thresholds, diagnostics |
| Simulated unlabeled pool | 25 | hidden-label candidate pool |
| Held-out test | 18 official-val logs | opened once |

Each window carries one of four label states per class — `positive`,
`negative`, `ignore` (event overlap without a bounded match), `invalid` — and
the last two are masked from loss; a Strem failure is never a negative. Public
pool records carry frame, scene, log, and time references only: no label,
binding, 3D box, or Strem result, and tests enforce that boundary.

## Code Map

| Module | Responsibility |
| --- | --- |
| `splits.py` | deterministic whole-log split |
| `projection.py` | 3D-box projection and the frozen eligibility filter |
| `strem_converter.py` / `strem_adapter.py` | pinned external Strem conversion and execution |
| `strem_eligibility.py` | keep only CAM_FRONT-learnable objects in streams |
| `scenario_events.py` / `window_dataset.py` | event grouping and five-frame window labels |
| `dino_features.py` | frozen DINOv2 feature cache |
| `lr_baselines.py` / `gru_baseline.py` | baselines and the ordered GRU |
| `false_negative_bank.py` / `public_pool.py` / `selection.py` | bad-case bank, label-free pool, frozen rankings |
| `oracle_reveal.py` / `feedback_retraining.py` | reveal selected IDs only; retrain declared arms |
| `prediction_scoring.py` | torch-free per-seed AP and seed-paired contrasts |
| `vlm_labeling.py` / `vlm_evaluation.py` | public-only VLM requests; scoring against the Oracle |

Twenty-three scripts under [`scripts/`](scripts) are the stage entry points;
stages chain by fixed artifact filenames and refuse to overwrite outputs.
Every number quoted in the docs traces to a JSON report under
[`results/`](results/README.md).

## Running It

```bash
uv sync --locked --all-groups        # add --all-extras for torch
uv run ruff check .
uv run mypy src scripts tests        # strict
uv run pytest                        # passes with no dataset present
```

- **Reading the results** needs nothing beyond this repository.
- **Tests** run from a bare clone; torch-dependent tests skip without the `ml`
  extra, and Strem integration tests skip unless `STREM_BIN` points at the
  pinned matcher.
- **Pipeline stages** need nuScenes, which is not redistributable — register at
  [nuscenes.org](https://www.nuscenes.org/). `v1.0-mini` metadata plus
  `CAM_FRONT` images are enough for a smoke run.
- **Full reproduction** is not possible outside the author's environment:
  scene mining calls Strem v0.3.0, an external research project whose
  repository is not public. Its binary is pinned by SHA-256 so a substituted
  build cannot silently change what a label means
  ([Strem Skill](docs/STREM_SKILL.md)). Everything after labeling is ordinary
  PyTorch and scikit-learn.

Details: [Architecture](docs/ARCHITECTURE.md) ·
[Data Spec](docs/DATA_SPEC.md) · [Evaluation Plan](docs/EVALUATION_PLAN.md) ·
[Findings](docs/FINDINGS.md)

## Scope

Core: scene mining, data cleaning, five-frame classification, bad-case mining,
fixed-budget data selection, controlled retraining, honest evaluation.
Not core: agents, RAG, SFT, vector databases, multi-sensor fusion,
planning/control, online learning, production claims.

## License

MIT — see [LICENSE](LICENSE). nuScenes data and the Strem binary are separate
works under their own licences and are not part of this repository.

## 中文简介

**自动驾驶场景挖掘与高价值数据闭环**：单人离线研究项目，在 nuScenes 上跑通并
逐环节实测一个数据闭环——形式化规则挖掘稀有时序场景、固定标注预算下盲选数据、
重训练、在从未接触过的整段行车日志上检验挖掘是否真的胜过随机。

| 问题 | 结论 |
| --- | --- |
| 相似度挖掘有效吗？(v0.8) | **无效。** 验证集显著（3.7σ）但 held-out 失效（0.3σ）；对照实验定位主因：选数据的依据与评估共用了行车日志 |
| 重新设计后闭环成立吗？(v0.10) | **成立。** 集成分歧选择在 10 段全新日志上胜过随机 `+0.062`（7.6σ，判据在揭示前冻结），三个预算档全正；14% 的新增标注拿到全量标注 85% 的增益（随机 55%） |
| VLM 自动标注能替代精确标签吗？(v0.11) | **此作业点上能。** F1 仅 0.49 的 VLM 标签拿到精确标签 86% 的训练收益，统计上不可区分，成本 `$0.026`/窗；学生模型加宽到 hidden 128 后走廊类仍成立、Macro 不成立（D5） |

仓库只含代码、场景规格、文档与聚合结果（`results/`），不含数据、图像、权重与
标签。测试无需数据集即可全部运行；完整复现需自行申请 nuScenes，且场景挖掘依赖
未公开的 Strem 匹配器（SHA-256 锁定）。完整实验叙事按证据到达顺序记录在
[docs/FINDINGS.md](docs/FINDINGS.md)，负结果全部保留。
