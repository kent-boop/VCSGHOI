# EvidenceHOI 示例图像挖掘：Codex 分阶段提示词包

## 0. 目标与约束

我要在 EvidenceHOI 论文中增加一张紧凑的机制可视化，核心不是展示“预测正确”，而是展示：

1. **Pair-specific evidence**：同一图像中的不同 human–object candidate 从共享视觉关系记忆中选择不同上下文证据；
2. **Hypothesis-sensitive matching**：固定同一 candidate pair，不同动作原型对同一视觉 evidence pool 产生不同的证据权重与匹配热力图；
3. **Decision relevance**：后续对最高权重 context path 做删除干预时，目标动作 logit 的下降应高于随机/低权重删除。

主实现版本：

```text
stip_graph6.py
visual_scene_graph_oov8.py
candidate_evidence_topology.py
text_scene_graph_oov3.py
text_prototype_slots.py
synchronized_evidence_topology_matcher.py
```

必须遵守：

- 所有导出功能默认关闭，不能改变训练、标准验证、checkpoint 兼容性或默认预测结果；
- 只在 `model.eval()` 且显式开启导出时记录；
- 不在模型 `forward()` 中直接写文件；模型只返回 trace，评测循环使用独立 writer 写盘；
- Graph6 的主线是 layerwise SETM。最终可视化数据必须来自最后一次有效的 `candidate_out_l / text_out_l / sync_out_l`，不能在 forward 末尾错误地重复执行一遍 final SETM；
- 同一 pair 的不同动作应共享同一 VET evidence pool；动作差异来自 SETM 的 prototype-conditioned matching/reweighting；
- `prototype_weights` 不是最终动作概率；最终动作概率来自原 action head 的 sigmoid；
- HOI 是多标签任务。竞争动作必须不是该 pair 的另一个 GT；
- `coarse_action_logits` 只能称作模型内部 coarse head，不能在论文中冒充独立训练的 baseline；
- 主文若绘制完整 `h→k→o` 两跳路径，优先筛选 `edge_hk=True AND edge_ko=True` 的 context；单侧有效只能标成 context evidence token。

---

# Prompt 0：先审计仓库，不修改代码

将下面提示直接发给 Codex：

```text
你现在是 EvidenceHOI 仓库的代码审计工程师。先不要修改任何代码。

目标：为论文机制可视化建立一个 evaluation-only 的 example-mining pipeline。请围绕以下文件审计：
- stip_graph6.py
- visual_scene_graph_oov8.py
- candidate_evidence_topology.py
- text_scene_graph_oov3.py
- text_prototype_slots.py
- synchronized_evidence_topology_matcher.py
- 实际 test/eval 入口、argument parser、dataset target 构造、postprocessor/evaluator

请逐项回答并给出准确文件名、类名、函数名、关键张量 shape：
1. Graph6 layerwise SETM 中，最后一层的 candidate_out_l、text_out_l、sync_out_l 在哪里产生，生命周期到哪里结束？
2. 当前 GraphReasonerOutput/debug_info 实际保留了哪些张量，哪些只有 shape/均值统计？
3. selected_context_indices、selected_context_mask、candidate evidence weights、selected_hoi_indices、prototype slots、text_to_visual_attn、SETM evidence_weights、prototype_coverage 的准确 shape 和含义是什么？
4. 最终 action logits、coarse sampled action logits、relation-existence logits 在 STIP forward 的哪里产生？
5. eval 时 targets 中实际有哪些键：image_id、file_name/path、orig_size、boxes、labels、relation_map 等？
6. DETR matcher 在 eval 中如何得到 detector-query 到 GT box 的映射？是否可复用 Criterion 的同一逻辑？
7. action id 的三套索引是否一致：
   a) action head logit index；
   b) args.valid_ids；
   c) text bank hoi_action_ids / hoi_ids 的 action id。
   若不一致，请明确映射方法。
8. HICO 图像路径如何由 target 或 image_id 解析？训练/测试文件名格式是什么？
9. 当前 eval 是否为 distributed；每个 rank 应如何无冲突地写 trace？
10. 当前 postprocessor 对 action score 做了哪些处理：sigmoid、pair threshold、pair-score weighting、object score multiplication、deduplication？

输出：
A. 一份“实施前审计报告”；
B. 最小改动文件列表；
C. 风险清单；
D. 不改默认行为的验收测试计划。

不要开始编码，直到上述索引映射和 eval 入口已经确认。
```

验收标准：Codex 必须确认实际 action-id 映射和真实 eval 入口；不能凭经验猜文件名或 target 字段。

---

# Prompt 1：在 visual_scene_graph_oov8.py 中导出最后一层 evidence trace

```text
基于上一阶段审计，修改 visual_scene_graph_oov8.py，为 Graph6 增加 evaluation-only evidence trace。

新增参数，默认值必须保持旧行为：
- graph_export_evidence_trace: bool = False
- graph_export_trace_each_layer: bool = False
- graph_export_trace_to_cpu: bool = True
- graph_export_trace_float_dtype: str = "float16"
- graph_export_trace_include_match_logits: bool = False
- graph_export_trace_include_features: bool = False

实现要求：
1. 在 layerwise SETM 路径中，每次生成 candidate_out_l、text_out_l、sync_out_l 后，若 export=True 且 not self.training，构建 compact trace；始终保存最后一次 trace，只有 each_layer=True 时才保留全部层。
2. 不要因为导出而额外执行 final SETM；Graph6 已通过 layerwise write-back 工作，最终 trace 应来自最后一次真实 layerwise matching。
3. trace 至少包含：
   - schema_version, stage, layer_idx
   - pair_indices, pair_valid_mask
   - selected_context_indices, selected_context_mask
   - candidate_evidence_mask, candidate_evidence_weights
   - selected_hoi_indices, selected_hoi_mask, selected_hoi_scores
   - selected_hoi_action_ids, selected_hoi_object_ids（从 text bank 映射，非法位置=-1）
   - prototype_slot_mask
   - sync_visual_to_text_attn
   - sync_text_to_visual_attn
   - sync_evidence_weights
   - sync_prototype_scores
   - sync_prototype_weights
   - sync_prototype_coverage
   - sync_anchor_delta
   - visual_token_roles = anchor/human/object/context_1...
   - prototype_slot_names = human_state/object_state/anchor_relation/discriminative/scene/context_object
4. 对每个 selected context k，从 edge_mask_dense 和 edge_score_dense 额外记录：
   - selected_context_edge_hk_mask
   - selected_context_edge_ko_mask
   - selected_context_two_hop_mask = hk & ko
   - selected_context_edge_hk_score
   - selected_context_edge_ko_score
5. 默认不要导出 D 维 features；只有 include_features=True 时才导出 evidence_tokens、prototype_slots、updated tokens。
6. 默认不要导出 raw match_logits；热力图使用归一化后的 text_to_visual_attn 即可。
7. 所有 tensor 必须 detach；浮点 tensor 按配置压缩；可选移到 CPU。不能保留 autograd graph。
8. 将最终 trace 放入 GraphReasonerOutput.debug_info["evidence_trace"]；each_layer=True 时再放 evidence_trace_layers。
9. 更新 build_visual_scene_graph_reasoner(args) 的 config 透传。
10. 对 non-layerwise/final matching 路径保持兼容：若该路径实际运行，也可导出同结构 trace。

安全性：
- export=False 时不得增加 GPU→CPU 同步、不得增加大 tensor 到 debug_info、不得改变数值路径；
- training=True 时即使误开 export 也不应导出；
- invalid pair、缺失 prototype、空 context 必须安全处理，无 NaN。

完成后：
- 运行 py_compile；
- 添加最小单元测试，构造 B=1,K=3,N=8 的假张量，验证 path support gather 和 trace shape；
- 输出修改摘要和测试命令。
```

---

# Prompt 2：在 stip_graph6.py 中补齐图像、节点、pair、预测和 GT 数据

```text
修改 stip_graph6.py，使 model.eval() 在 graph_export_evidence_trace=True 时返回 outputs["evidence_traces"]，一张图一个 record；默认关闭时完全不改变旧输出和性能。

新增/读取参数：
- graph_export_evidence_trace=False
- graph_export_trace_float_dtype="float16"
- graph_export_trace_topk_actions=10
- graph_export_trace_include_all_action_logits=True
- graph_export_trace_match_ground_truth=True

每张图的 record schema：

record = {
  "schema_version": 1,
  "image": {
    "image_id": ...,
    "file_name": ...,
    "orig_size_hw": [H,W],
    "batch_image_index": ...
  },
  "nodes": {
    "boxes_xyxy_norm": [N,4],
    "scores": [N],
    "labels": [N],
    "types": [N],
    "valid_mask": [N]
  },
  "pairs": {
    "indices": [K,2],
    "proposal_logits": [K],
    "proposal_probs": [K],
    "object_labels": [K],
    "object_scores": [K],
    "action_logits": [K,A],
    "action_probs": [K,A],
    "ranking_action_scores": [K,A],
    "action_topk_ids": [K,T],
    "action_topk_probs": [K,T],
    "coarse_action_logits": [K,A],
    "coarse_action_probs": [K,A],
    "gt_action_targets": uint8 [K,A],
    "gt_pair_matched": bool [K]
  },
  "graph": graph_out.debug_info["evidence_trace"] 去掉 B=1 维,
  "metadata": {
    "dataset_file": ...,
    "valid_action_ids": ...,
    "valid_object_ids": ...,
    "pair_filter_threshold": ...,
    "use_pair_score_weighting": ...,
    "graph_trace_available": ...
  }
}

实现细节：
1. final action logits 使用 decoder 最后一层 action_embed 输出，不要使用 prototype_weights 代替。
2. coarse_action_logits 使用 sampled_actions；在文档中明确它是内部 coarse head，不是独立 baseline。
3. eval 且 match_ground_truth=True 时，复用同一个 DETR matcher 获取 det query→GT box 映射，再按 Criterion 现有逻辑从 target['relation_map'] 生成每个 sampled pair 的多标签 GT 向量。
4. 若 pair 两端没有同时匹配 GT，gt_pair_matched=False，GT 向量保持 0；不要把未匹配 pair 当成真负例用于主图筛选。
5. ranking_action_scores 要复现 postprocessor 在 dedupe 前的主要 score 逻辑，并同时保留 raw action_probs：
   - sigmoid(action_logits)
   - object score
   - pair_filter_threshold
   - 可选 pair-score weighting
   对 dedupe 无需在 trace 中重做，但要在 metadata 中注明 pre-dedupe。
6. node 数组保留原 DETR query index，因为 selected_context_indices 和 pair_indices 都引用该索引。
7. target 路径字段须根据审计结果读取；实现兼容候选键，但不要伪造路径。
8. graph trace 已经 detach/CPU；STIP 新增 tensor 也要 detach/压缩/CPU。
9. 不要在 forward 中 torch.save。
10. outputs 中只有 export=True 且 eval 时才新增 evidence_traces。

测试：
- export=False：比较固定输入下旧版与新版所有预测 tensor bitwise/近似一致；
- export=True：检查 record schema、tensor shape、无 requires_grad、全部 CPU；
- multi-label pair：GT 中两个 action 均为 1；
- unmatched pair：gt_pair_matched=False；
- batch size > 1：每图 trace 不串位。
```

---

# Prompt 3：在 eval loop 中使用独立 writer

```text
新增 src/.../evidence_trace_io.py（放置位置遵循仓库结构），实现 EvidenceTraceWriter，并接入真实 test/eval loop。

原则：模型不写文件，评测循环在 outputs 返回后调用 writer.write_outputs(outputs)。

Writer 要求：
1. 每张图保存独立 .pt：records/{image_id}_r{rank}_{counter}.pt；先写 .tmp 再原子 rename。
2. 每个 distributed rank 写 manifest_rank{rank}.jsonl，避免并发写同一文件。
3. manifest 只保存轻量摘要：record_path、image_id、file_name、pair_count、positive_pair_count、mean_context_count、mean_two_hop_count、max_action_prob、min_top2_margin。
4. 提供 merge_manifests() 合并每个 rank 的 manifest。
5. 可写 run_metadata_rank{rank}.json，至少包含 checkpoint、git commit、完整关键配置、action/object name map 来源、dataset split、随机种子。
6. 支持断点续跑与 overwrite 选项。
7. writer 只在 args.graph_export_evidence_trace=True 时构建；普通评测不创建目录、不写文件。

请在 argument parser 中增加：
--graph-export-evidence-trace
--graph-export-trace-dir PATH
--graph-export-trace-each-layer / --no-...
--graph-export-trace-to-cpu / --no-...
--graph-export-trace-float-dtype {float16,float32,bfloat16}
--graph-export-trace-include-match-logits
--graph-export-trace-include-features
--graph-export-trace-topk-actions INT
--graph-export-trace-include-all-action-logits / --no-...
--graph-export-trace-match-ground-truth / --no-...
--graph-export-trace-overwrite

请基于仓库真实 parser 风格实现，不要创建第二套互不连通的 parser。

评测循环伪代码：
writer = None
if args.graph_export_evidence_trace:
    writer = EvidenceTraceWriter(args.graph_export_trace_dir, rank=get_rank(), overwrite=...)
...
outputs = model(samples, images, targets)
if writer is not None:
    writer.write_outputs(outputs)
...
barrier()
if rank == 0:
    merge manifests

注意：如果 evaluator 会把 outputs 跨 GPU gather，先在本 rank 写 trace，并在送入 evaluator 前排除 evidence_traces，避免大对象 all_gather。

测试：单 GPU 10 张图、2 GPU 各 10 张图、重复运行、空 trace、磁盘中断后的临时文件清理。
```

---

# Prompt 4：实现自动筛选脚本，输出三个排行榜

```text
实现 tools/rank_evidence_examples.py，读取 trace_dir/records/*.pt，不运行模型，只做离线筛选。

必须输出：
1. hypothesis_sensitive_candidates.csv
2. pair_specific_candidates.csv
3. combined_hero_candidates.csv
4. ranking_summary.json

A. 硬过滤（默认参数可调）：
- pair_valid=True；
- gt_pair_matched=True；
- 至少 2 个 selected context；
- 至少 1 个 selected context 满足 hk AND ko；
- context detection mean score >= 0.15；
- context median normalized box area >= 0.002；
- GT action 被检索进 selected prototypes；
- competitor action 不属于该 pair 的任何 GT action；
- 最终 p_gt > p_comp（默认要求，可关闭）；
- pair-specific 排行默认要求 controlled comparison：same object node > same human node > same object class。

B. Hypothesis-sensitive 指标：
- gt action 和 strongest non-GT retrieved prototype；
- raw final action probability 与 logit；
- ambiguity：两动作都具有一定概率且分数接近；
- JS(sync_evidence_weights_gt, sync_evidence_weights_comp)；
- prototype-slot heatmap distance：对 text_to_visual_attn 按共同有效 slots 计算平均 total-variation distance；
- evidence concentration = 1 - normalized entropy；
- top-context switch；
- prototype coverage；
- refinement gain：
  G = (z_gt-z_comp)_final - (z_gt-z_comp)_coarse
  并明确 G 仅为模型内部 refinement mining 指标，不是独立 baseline 对比。

C. Pair-specific 指标：
对同图两个 pair 的 selected context node 集计算：
- set Jaccard distance；
- 以 CETI context weights 为权重的 weighted Jaccard distance；
- controlled-comparison bonus：same object node=最高，same human 次之，same object class 再次。

D. Visual quality：
- human/object endpoint 最小框面积；
- context 中位框面积；
- context detection score；
- valid node 数过多的 clutter penalty；
- 横向/4:3 图像比例偏好。

E. 排行策略：
不要只输出一个总分。分别保留 pair-specific 和 hypothesis-sensitive 排行，再做 combined hero 排行。总分只用于 mining，不作为论文实验指标。

建议组合：
- hypothesis_score = evidence divergence + heatmap divergence + ambiguity + concentration + top-context-switch + confidence + refinement gain；
- pair_specific_score = set/weighted distance + control bonus + visual/path quality；
- combined_score = 0.38*hypothesis + 0.30*pair-specific + 0.14*refinement + 0.10*visual + 0.08*path。

脚本必须支持 action_names_json、object_names_json，以便 CSV 直接显示名称；若缺失则显示数字 id。

编写 synthetic unit test，构造一张 3-pair 图，确保：
- multi-label GT 不会被误当 competitor；
- same-object-node pair 排名高于 uncontrolled pair；
- evidence distribution 越不同，hypothesis score 越高；
- tiny/low-score context 被过滤。
```

---

# Prompt 5：生成候选图检查面板

```text
实现 tools/render_evidence_shortlist.py，输入 combined_hero_candidates.csv、trace records、image_root，输出前 N 个 PNG 检查面板。

每个面板：
- 左：原图叠加 focal pair 的 human/object 框、Top context 框；若有 partner pair，使用另一组线型/颜色显示；
- 右上：GT action 的 text_to_visual_attn 热力图，行 H/O/A/D/S/C，列 Anchor/Human/Object/C1...；下方画 evidence weight strip；
- 右下：competitor action 的同结构热力图；
- 标题显示 image_id、pair_idx、gt/competitor、combined score、pair control type；
- 图像路径优先使用 trace.image.file_name；支持 image_root + image_pattern fallback；
- 缺图写 missing_images.txt，不要中断整个任务。

这只是人工筛选面板，不是投稿终稿图。保持代码无外部服务依赖。
```

---

# Prompt 6：对前 20–50 张候选做第二遍证据删除验证

```text
现在只对第一遍 ranking 的前 20–50 个候选实现 second-pass faithfulness test。

目标：比较 Original、Remove Top Context、Remove Random Context、Remove Bottom Context 的目标动作 logit。

要求：
1. 只删除 context tokens，不删除 anchor/human/object；
2. 删除后重新归一化 evidence mask/weights，并重新执行受影响的 SETM 与后续 prediction；
3. 若 Graph6 是 layerwise write-back，优先实现 forced_context_exclusion，在每个同步层都屏蔽同一个 detector node；
4. 若暂时只能屏蔽最后一层，输出字段和论文表述必须明确为 final-layer evidence masking；
5. random deletion 至少 5 个固定 seed，记录 mean±std；
6. 保存：target action logit/prob、competitor logit/prob、prediction flip、被删除 node id/box/class、每个条件的 trace；
7. 不要把 deletion 结果称作“因果证明”，使用 decision-relevant / faithfulness diagnostic。

建议验收条件：
Delta_top > Delta_random_mean > Delta_bottom
并优先选择差距稳定、图像语义清晰的例子进入主文。
```

---

# Prompt 7：最终代码审查

```text
请对本次 example-mining 改动做最终审查，不再增加新功能。

检查：
1. 默认 flags 全 False 时，训练与测试数值路径是否完全不变；
2. 是否有任何 trace tensor 在 export=False 时被构造或 CPU 同步；
3. 是否错误地在 layerwise SETM 后重复 final matching；
4. 是否把同一 pair 不同 action 画成不同 VET node selection；
5. action id / valid_ids / text-bank id 映射是否经过测试；
6. GT multi-label competitor 排除是否正确；
7. distributed writer 是否会冲突或把大 trace all_gather；
8. trace 文件是否包含 checkpoint/config/git hash，可复现；
9. 是否有 NaN、invalid index、空 context、缺 prototype 的测试；
10. 是否提供 README，包括扫描、排行、渲染、second-pass 的完整命令。

最后输出：
- 改动文件清单；
- schema 文档；
- 运行命令；
- 测试结果；
- 已知限制；
- 论文中允许和不允许的表述。
```

---

# 推荐运行流程

## 第一遍：轻量全测试集扫描

```bash
python <your_eval_entry>.py \
  --resume /path/to/final_checkpoint.pth \
  --eval \
  --graph-export-evidence-trace \
  --graph-export-trace-dir outputs/evidence_trace \
  --graph-export-trace-float-dtype float16 \
  --graph-export-trace-topk-actions 10 \
  --graph-export-trace-include-all-action-logits \
  --graph-export-trace-match-ground-truth \
  --no-graph-export-trace-each-layer \
  --no-graph-export-trace-include-features \
  --no-graph-export-trace-include-match-logits
```

建议第一遍只保存最后一层注意力和权重，不保存 D 维 feature，不保存 raw match logits。

## 离线排名

```bash
python tools/rank_evidence_examples.py \
  --trace-dir outputs/evidence_trace \
  --output-dir outputs/evidence_ranking \
  --action-names-json assets/action_names.json \
  --object-names-json assets/object_names.json \
  --top-n 200 \
  --min-context-count 2 \
  --min-two-hop-count 1 \
  --min-context-score 0.15 \
  --min-context-area 0.002 \
  --require-gt-prototype \
  --require-final-correct \
  --require-controlled-pair
```

## 渲染前 50 张检查面板

```bash
python tools/render_evidence_shortlist.py \
  --ranking-csv outputs/evidence_ranking/combined_hero_candidates.csv \
  --image-root /path/to/hico/images/test2015 \
  --output-dir outputs/evidence_panels \
  --limit 50 \
  --action-names-json assets/action_names.json
```

## 人工复核

优先保留：

- 同一 object node 或同一 human 下的两个 pair，selected context 差异清楚；
- GT vs competitor 都语义合理，但 SETM 热力图/权重确实不同；
- context 对象检测准确、框足够大；
- 至少一条真正两跳 path；
- 有一个显眼但未选择的实体可作为“未进入证据链”的定性干扰项，但不要声称模型显式证明它无关；
- coarse head 犹豫、final head 改善，可作为筛选依据；
- second-pass Top deletion 明显大于 Random/Bottom。

最终主文只需 1 张 hero image 或最多 2 张原图；其余放 supplementary。
