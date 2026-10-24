# EvidenceHOI Trace 与论文可视化任务交接书

更新时间：2026-07-13  
项目目录：`/wzj/zijian/OSSM-HOI-main`

## 1. 一句话状态

EvidenceHOI Graph6 M4/R5 的完整测试集推理和 Trace 全链路已经跑通，9546 张测试图的 Trace 已落盘并通过校验；当前不需要重新跑模型。尚未完成的是从 Trace 中筛出真正适合论文展示的高质量证据链图片。前几轮候选已被用户否定，下一步应改为优先挖掘“正确 GT、上下文证据较强、上下文不是目标物重复框、画面清晰”的单 pair 案例，而不是强行寻找同图双 pair。

## 2. 用户最终目标

从训练好的 EvidenceHOI 模型测试结果中选择 3-5 张适合论文可视化的图片，展示：

- Human-Object pair；
- 模型选中的上下文节点；
- H-C、C-O 边和 one-hop/two-hop 状态；
- 检索到的动作 prototype；
- A/H/O/C1-C4 的最终证据分配；
- 正确预测、易混淆动作或失败案例之间的区别。

图片必须优先满足：主体清楚、框准确、HOI 类型有差异、上下文物体具有真实语义差异，并且结论不超出 Trace 数值本身。

## 3. 运行环境与模型配置

- Conda 环境：`SPLHOI`
- 数据集：HICO-DET test2015
- 测试图目录：`/share/share/Datasets/hico_20160224_det/images/test2015`
- DETR backbone：ResNet-101
- DETR 权重：`/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth`
- BLIP：Large
- BLIP 权重：`/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth`
- Graph6 配置：M=4，R=5
- 本次完整 Trace 实际使用的 checkpoint：
  `/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth`
- 用户最初指定的另一个 checkpoint：
  `/wzj/zijian/OSSM-HOI-main/checkpoints/EvidenceHOI-Large/STIP_opensource/hico_run_graph6_base/best.pth`

注意：最初 checkpoint 在 M4/R5 配置下出现 shape mismatch：

- `rank_embedding.weight`：checkpoint `[3,256]`，当前模型 `[5,256]`
- `role_embedding.weight`：checkpoint `[4,256]`，当前模型 `[7,256]`

因此完整 M4/R5 Trace 使用了与该配置匹配的 `hico-det_sync_vt_graph6` checkpoint。不要直接把不匹配参数静默裁剪后作为正式结果。

## 4. 完整测试与 Trace 状态

- 完整测试已成功完成。
- 实测 mAP：`48.7473`
- 唯一测试图 Trace：`9546`
- 重复记录：`6`，已单独归档。
- Trace 总目录：
  `/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_graph6_m4r5_full/traces`
- 唯一记录目录：
  `/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_graph6_m4r5_full/traces/records`
- 合并清单：
  `/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_graph6_m4r5_full/traces/manifest.jsonl`
- 校验汇总：
  `/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_graph6_m4r5_full/traces/validation_summary.json`
- Trace 约占 `1.2G`。

历史上 Trace 先写到了容器 `/tmp/evidencehoi_graph6_m4r5_20260712_204829/traces`，后来已经复制到项目输出目录。容器没有 `rsync`，后续不要依赖它。

## 5. 相关脚本与代码

### 推理与 Trace

- `run_evidence_trace_graph6_m4r5_v2.sh`
- `src/models/stip_graph6.py`
- `src/models/stip_evidence_trace.py`
- `src/models/evidence_trace_io.py`
- `tools/evidence_vis/validate_evidence_trace.py`
- `tests/test_evidence_trace_chain.py`

### 候选筛选与可视化

- `tools/evidence_vis/rank_evidence_examples.py`
  - 已支持 GT action 字段。
  - 已增加 `--pair-require-gt-matched`、`--pair-require-final-correct`。
  - combined 排序已修复“先截断再组合”问题。
- `tools/evidence_vis/render_pair_specific_shortlist.py`
  - 已增加受控端点 IoU 过滤。
- `tools/evidence_vis/rank_contrastive_evidence_chains.py`
  - 新增的双正 GT 对比排序器。
- `tools/evidence_vis/rank_anchor_partner_evidence.py`
  - 新增的“正确 GT 锚点 vs 不同 partner”流式排序器。
- `tools/evidence_vis/render_evidence_chain_examples.py`
  - 完整证据链面板渲染器。
  - 面板包含 pair、context、边强度、prototype 和证据分配。

截至最后一次检查，上述三个新脚本仍是 Git 未跟踪文件。不要在未检查工作区的情况下清理或覆盖它们。

## 6. 标签映射

- 动作名称：
  `outputs/evidence_trace_graph6_m4r5_full/label_maps/action_names.json`
- HICO 80 类物体名称：
  `outputs/evidence_trace_graph6_m4r5_full/label_maps/object_names.json`
- detector/COCO 节点名称：
  `outputs/evidence_trace_graph6_m4r5_full/label_maps/coco_node_names.json`

pair 的 object label 与 graph node 的 detector label 不属于同一个编号空间，渲染或筛选时必须分别使用对应映射。

## 7. 已完成的数据分析

### 正 GT 数量

- 9546 张 Trace 中，至少有 1 个正 GT pair 的图片：`715`
- 至少有 2 个正 GT pair 的图片：`93`
- 93 张图中，存在不同正 GT action 的图片：`18`
- 不同正 GT action 的 pair 组合：`33`

轻量符号链接子集：

- 至少 1 个正 GT：
  `outputs/evidence_trace_graph6_m4r5_full/gt_anchor_scan`
- 至少 2 个正 GT：
  `outputs/evidence_trace_graph6_m4r5_full/gt_pair_scan`

这些目录主要是符号链接，不是重复复制 Trace。

### 双正 GT 对比结论

严格要求“两个 pair 都是正 GT、GT 动作不同、上下文类别不同、端点不重复”时，仅找到少量 hot-dog `cook/make` 组合；没有找到“两边都预测正确”的合格案例。

原因不是没有不同动作，而是多数不同动作案例不满足旧规则中的 `每组至少一个 two-hop`。已确认存在的动作组合包括：

- `hold / carry / stand_under`
- `ride / sit_on`
- `ride / straddle`
- `stand_on / jump`
- `hold / pick_up`
- `lie_on / hold`
- `cook / make`

但很多图片的上下文仍是目标物的重复检测，画面或语义质量不适合作为论文主图。

### Hypothesis-sensitive 结论

- 64 个 hypothesis 候选中仅 `1` 个发生 top-context switch。
- 最大 JS divergence：约 `0.0007102`
- 最大 heatmap distance：约 `0.0010254`

因此当前数据不支持“不同动作假设显著改变上下文分配”的强结论。

### 最终上下文证据占比

已渲染案例中，action-specific context 总权重通常只有约 `0.013%-0.039%`，Human/Object 槽占据绝大部分最终证据。

可视化可以支持：

- pair-specific context selection；
- prototype retrieval；
- one-hop/two-hop 路径形成；
- 动作混淆或预测坍缩诊断。

暂时不能支持：

- “上下文节点主导最终预测”；
- “证据链显著提升该单样本预测”；
- “不同动作假设产生大幅不同的最终证据分配”。

## 8. 已生成但未获用户认可的结果

### 旧版

- `outputs/evidence_trace_graph6_m4r5_full/evidence_chain_examples_v2`
- bird/feed、cow/herd、hot-dog cook/make。

用户反馈：前两张 HOI 类型相同，上下文物体基本相同，不能体现证据链效果；hot-dog 稍好。

### 新版但仍被否定

- `outputs/evidence_trace_graph6_m4r5_full/evidence_chain_examples_v3_contrastive`
- 包括 bench/cell-phone、table/TV、chair/book、table/pizza、hot-dog。

用户反馈：这几张图整体效果仍不好。

主要问题：

- 4 张依赖 false-positive partner，论文主图说服力不足；
- 部分画面杂乱或上下文检测类别不可靠；
- 仍然是为了满足双 pair 对比而牺牲了画面质量；
- context 最终权重很小。

不要继续在 v3 上小修小补，应换筛选策略。

## 9. 下一步正确策略

不要再强行要求同图双 pair。先从 715 张至少有一个正 GT pair 的图片中筛选单个高质量正确 pair，再将 3-5 个不同 HOI 类型的单 pair 案例组成论文图组。

每个候选优先满足：

1. `gt_action_targets[pair].any() == True`
2. top-1 action 属于 GT actions
3. 使用 `sync_evidence_weights` 中对应预测 action prototype 的证据分配
4. context action weight 总和处于全局高位
5. 至少一个 informative context：
   - 类别不是 `person`
   - 类别不是当前目标物
   - context box 与 H/O endpoint 最大 IoU 小于约 `0.65`
6. informative context 自身最好具有 two-hop 路径
7. context detector score、pair probability 和 visual quality 较高
8. 人、物体、上下文在原图中肉眼可辨，框不遮挡主体
9. 最终选出的图片覆盖不同 HOI，例如 locomotion、object manipulation、sports、household interaction 等

建议记录以下指标：

- `context_weight_sum`
- `informative_context_weight_sum`
- `context_weight_percent`
- `informative_context_weight_percent`
- context 类别、检测分数、endpoint IoU
- context 是否 two-hop
- pair top-1 概率
- visual quality

## 10. 下一步执行顺序

1. 新建或完成：
   `tools/evidence_vis/rank_correct_context_evidence.py`
2. 扫描：
   `outputs/evidence_trace_graph6_m4r5_full/gt_anchor_scan/records`
3. 输出新排名，建议目录：
   `outputs/evidence_trace_graph6_m4r5_full/ranking_v14_correct_context`
4. 查看 context weight 的 p50、p90、p99、最大值，先确认是否存在明显高于旧案例的样本。
5. 从不同 HOI 类别各取若干候选，渲染约 20-30 张简洁预览。
6. 人工检查原图和框，删除重复检测、错误类别、画面过暗和主体过小的案例。
7. 只保留 3-5 张真正清晰的正确 GT 图片。
8. 新建结果目录，例如：
   `outputs/evidence_trace_graph6_m4r5_full/evidence_chain_examples_v4_correct_gt`
9. 不覆盖 v2、v3，方便回溯。
10. 运行：
    `/share/apps/anaconda3/envs/SPLHOI/bin/python tests/test_evidence_trace_chain.py`

## 11. 运行注意事项

- 不需要重新跑测试模型，优先使用已有 Trace 做离线筛选。
- 不要修改或覆盖 checkpoint。
- 如确需改动模型代码，应新建文件或先存档旧版本。
- 不要删除用户现有代码或未提交改动。
- 当前工作区可能包含大量未跟踪文件，提交前先检查 `git status`。
- 不要把 checkpoints、可视化大图、`.npy`、Trace `.pt` 等大文件提交到 GitHub。
- 容器没有 `rsync`。
- 此前 Codex 权限审批后端发生过超时；这是工具环境问题，不是项目代码错误。

## 12. 新话题建议开场白

可在新话题中直接发送：

> 请先阅读 `/wzj/zijian/OSSM-HOI-main/EVIDENCE_TRACE_VISUALIZATION_HANDOFF_2026-07-13.md`，然后继续 EvidenceHOI 证据链图片筛选任务。不要重新跑模型，不要覆盖 checkpoint 或旧可视化结果。请从 715 张含正 GT 的 Trace 中优先筛选 top-1 正确、informative context 证据占比较高、上下文不是目标物重复框、画面清晰的单 pair 案例，人工检查后生成 3-5 张新的论文级证据链图，并如实报告 context 最终权重。

## 13. 完成标准

本任务只有同时满足以下条件才算完成：

- 3-5 张图片均为正确 GT pair 主案例；
- HOI 类型有明显多样性；
- context 节点肉眼可辨且不是简单重复框；
- 可视化包含完整证据链字段；
- 每张图都有可核对的数值表；
- 对 context 权重很小等负面结果不隐瞒；
- 用户目视认可图片质量。
