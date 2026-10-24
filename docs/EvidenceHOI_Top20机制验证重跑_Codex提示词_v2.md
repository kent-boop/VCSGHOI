# EvidenceHOI 固定 Top-20 机制验证重跑：Codex 执行提示词（服务器实际环境版 v2）

> 用途：补齐论文补充材料第四章的逐层机制动态，以及第五章成功案例筛选所需的因果对照、GT 映射和 provenance 数据。
>
> 本提示词已经按照当前服务器上的真实仓库、模型入口、checkpoint、旧 Top-20 trace 和现有脚本进行调整。执行者不得再把任务理解为全测试集评测，也不得只输出建议而不实际实现、运行、验证和打包。

---

## 0. 执行角色与最终目标

你正在服务器上的 EvidenceHOI 项目中工作。请完成一次“固定 Top-20 机制验证任务”，实际检查代码、补充只读 trace、运行固定 Top-20 推理、执行确定性机制对照、验证所有输出，并生成自包含压缩包。

最终交付只负责生成可靠、可复现、可直接绘图的数据，不绘制投稿图。

除非以下关键资产在服务器上确实不存在，否则不要停下来询问用户：

- checkpoint；
- 固定 Top-20 清单；
- HICO-DET test annotation；
- prototype bank；
- BLIP 或 DETR 权重。

如果普通字段命名存在轻微歧义，优先采用本提示词给出的字段语义，同时额外保存 raw ID、model-local ID 和解释文档，不要因可通过兼容字段解决的问题中断任务。

---

## 1. 已确认的服务器资产

### 1.1 仓库与模型

仓库根目录：

~~~text
/wzj/zijian/OSSM-HOI-main
~~~

最终用于本任务的 checkpoint：

~~~text
/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth
~~~

DETR 权重：

~~~text
/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth
~~~

BLIP 权重：

~~~text
/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth
~~~

基础配置：

~~~text
BLIP_VIT=large
BACKBONE=resnet101
M=4
K=5
graph_num_layers=2
num_hoi_queries=32
dataset=hico-det
split=test
~~~

### 1.2 数据与 prototype 资产

HICO-DET 数据集：

~~~text
/share/share/Datasets/hico_20160224_det
~~~

测试标注：

~~~text
/share/share/Datasets/hico_20160224_det/annotations/test_hico.json
~~~

测试图片：

~~~text
/share/share/Datasets/hico_20160224_det/images/test2015
~~~

OO prior：

~~~text
/wzj/zijian/OSSM-HOI-main/oo_affordance_prior_80.npy
~~~

Prototype bank 与索引：

~~~text
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_text_features_6x256.npy
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_text_mask_6.npy
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_hoi_ids.json
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_object_to_hoi_indices.json
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_field_order.json
/wzj/zijian/OSSM-HOI-main/hico_text_graph_blip256/hico_hoi_graph_blip256_prompt_metadata.json
~~~

### 1.3 固定 Top-20 与旧参考结果

固定 Top-20 原始排名清单：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_stip_final_20260731_190715/supplementary_top20_mechanism_relative/ranking/top20_mechanism_candidates.csv
~~~

上一轮详细 Top-20 输出：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed
~~~

上一轮详细 trace：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed/traces
~~~

上一轮 pair remapping：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed/top20_detailed_pair_mapping.csv
~~~

上一轮本地数据包：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed/local_vet_setm_package_detailed
~~~

固定 Top-20 不得重新排名或替换。必须沿用原始 20 个 candidate_id、image_id、原始 human/object boxes 和 original_pair_idx。

### 1.4 现有运行入口

已有 Top-20 入口：

~~~text
/wzj/zijian/OSSM-HOI-main/run_top20_detailed_vet_setm_trace.sh
~~~

已有基础 trace 入口：

~~~text
/wzj/zijian/OSSM-HOI-main/run_evidence_trace_stip_final.sh
~~~

已有验证和导出脚本：

~~~text
/wzj/zijian/OSSM-HOI-main/tools/evidence_vis/validate_detailed_vet_setm_trace.py
/wzj/zijian/OSSM-HOI-main/tools/evidence_vis/remap_ranking_pairs_to_trace.py
/wzj/zijian/OSSM-HOI-main/tools/evidence_vis/export_vet_setm_local_package.py
~~~

---

## 2. 当前代码的真实入口与修改边界

必须先确认而不是猜测模型入口。

当前运行参数：

~~~text
--stip_model_variant evidence_trace
~~~

实际由 src/models/detr.py 加载：

~~~text
src/models/stip_evidence_trace.py
~~~

不是直接加载 src/models/stip_graph6.py。

当前 src/models/stip_evidence_trace.py 与 src/models/stip_graph6.py 的模型主体基本一致，已知主要差异是 GT trace 对 relation_map/hois 的兼容处理。因此：

1. 新增的 evaluation-only query/probe trace 必须首先在 src/models/stip_evidence_trace.py 生效；
2. 对属于 Graph6 公共逻辑的修改，应同步到 src/models/stip_graph6.py，或者抽成共享 helper，避免两个文件继续分叉；
3. 不要为了本任务修改训练语义；
4. 默认不要修改 src/models/stip.py；
5. 所有新增功能必须由显式 debug/trace flag 控制，默认关闭；
6. flag 关闭时，训练和普通评测行为必须完全保持不变；
7. 不更新 checkpoint，不写入 prototype bank，不训练任何参数。

视觉图和 SETM 的主要实际实现：

~~~text
src/models/visual_scene_graph_oov8.py
src/models/visual_evidence_topology_vet_layerwise_setm.py
src/models/synchronized_evidence_topology_matcher.py
src/models/text_scene_graph_oov3.py
src/models/text_prototype_slots.py
~~~

执行前生成一份 reports/code_audit.md，列出：

- 实际 import 链；
- query residual 的真实代码位置；
- layer loop 的真实代码位置；
- retrieval 初始化位置；
- SETM 调用位置；
- visual/text write-back 位置；
- 最终 pair_out_mlp 位置；
- final/post-hoc SETM 是否被跳过；
- 新增 probe 的插入位置；
- 任何与本提示词算法约束不一致的代码。

如发现算法冲突，先记录具体文件、行号和影响。不得静默改变算法；只有 trace/debug 行为可以直接修正。

---

## 3. “一次验证任务”的精确定义

本任务不重跑 9,546 张完整测试集，不训练模型。

它是一个统一的 Top-20 验证任务，内部至少包含 5 个确定性 variant：

1. full；
2. no_anchor_writeback；
3. no_prototype_state_writeback；
4. no_setm；
5. endpoint_only_setm。

因此：

- full 只跑固定 20 张图片；
- 其余每个 variant 也只跑相同的 20 张图片；
- 可用一个统一 wrapper 顺序执行并汇总；
- 不得把对照结果从 full 的参数或 trace 中数学推算出来；
- 不得重新进行全测试集 mAP 评测；
- 不要求重新挑选 Top-20；
- 可选的随机 donor 实验默认不运行。

上一轮 Top-20 推理的模型推理时间约 29 秒，因此本任务重点是 trace 正确性与验收，而不是全测试集计算。

允许先用固定 Top-20 中的 1 张图片做实现 smoke test。smoke 结果写入独立临时目录，不作为正式数据；正式结果仍必须完整跑 20/20。

---

## 4. 不可改变的算法事实

### 4.1 VET

每个 candidate human-object pair 的证据顺序固定为：

~~~text
V_q^l = [E_ho^l, X_h^l, X_o^l, P_1^l, P_2^l, P_3^l, P_4^l]
~~~

因此 M=4 时：

~~~text
visual unit count = 3 + M = 7
~~~

视觉角色顺序必须固定：

~~~text
E_ho, X_h, X_o, P_1, P_2, P_3, P_4
~~~

### 4.2 Prototype roles

每个 prototype 固定包含 6 个 role：

~~~text
H, O, A, D, S, C
~~~

含义：

~~~text
H = Human State
O = Object State
A = Anchor Relation
D = Discriminative Cue
S = Scene Cue
C = Context Object
~~~

必须同时保存：

- canonical role：H/O/A/D/S/C；
- raw role name；
- role index；
- valid mask；
- prompt 原文；
- text bank index；
- prototype HOI/action/object ID。

### 4.3 SETM

当前配置下，每个 prototype 的 correspondence matrix 为：

~~~text
7 × 6
~~~

完整 match_logits 还包含 Top-5 prototype 维度。

两个方向必须来自同一个 match_logits：

- visual→text：对 text-role 维 softmax；
- text→visual：对 visual-unit 维 softmax。

禁止独立构造两个不一致的匹配矩阵。

### 4.4 Layerwise 标准路径

必须保持：

- retrieval 只在 layer loop 前执行一次；
- graph_text_dynamic_retrieval=False；
- Top-5 prototype IDs 在两层间不变；
- 每层最多一次 SETM anchor correction；
- graph_layerwise_sync_final_matching=False；
- layer loop 后不得再执行 post-hoc/final SETM；
- graph_layerwise_sync_disable_old_dense_injection=True；
- Frozen Prototype Bank 只读；
- prototype write-back 只更新 candidate-local temporary text state；
- 正确或 GT HOI 类别不得进入 retrieval；
- prediction 始终是 Base HOI Query 加 gated graph correction 后进入原 HOI decoder。

### 4.5 只读 probe

所有新增 probe 必须：

- model.eval()；
- torch.no_grad()；
- 不写参数；
- 不反馈到正常 forward；
- 不改变 normal final logits；
- 不改变 proposals、pair order、retrieval 或 prototype bank；
- 使用原 interaction_decoder 和 action_embed；
- 使用与 normal forward 完全一致的 memory、positional encoding、layout mask、dependency encoding 和 proposal order。

---

## 5. 新输出与历史结果隔离

新输出目录使用唯一 timestamp，例如：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_mechanism_validation_v2_YYYYMMDD_HHMMSS
~~~

不得覆盖：

- 旧全测试集 trace；
- 旧 Top-20 trace；
- 旧 local package；
- 旧可视化；
- checkpoint；
- prototype bank；
- 原始附件。

所有写入先进入当前 run 的临时文件，完成后原子重命名。支持断点续跑：

- 每个 variant 有独立目录和完成标志；
- 只有在该 variant 验证通过后才写完成标志；
- 续跑前先重新校验已有输出；
- 不以“文件存在”代替“验证通过”。

---

## 6. Phase 0：冻结运行元数据

正式加载模型和推理前，生成：

~~~text
run_meta.json
config/full_args.json
config/environment.json
config/file_hashes.json
config/variant_config_diff.json
reports/code_audit.md
~~~

run_meta.json 至少包含：

~~~text
run_id
timestamp_local
timestamp_utc
hostname
repository_root
git_commit
git_status_dirty
git_status_short
checkpoint_path
checkpoint_sha256
checkpoint_epoch
detr_weights_path
detr_weights_sha256
blip_pretrained_path
blip_pretrained_sha256
dataset_name
dataset_split
dataset_root
dataset_annotation_path
dataset_annotation_sha256
image_root
top20_mapping_path
top20_mapping_sha256
old_detailed_trace_path
prototype_bank_path
prototype_bank_sha256_before
prototype_mask_path
prototype_mask_sha256
prototype_prompt_metadata_path
prototype_prompt_metadata_sha256
action_label_source
action_label_sha256
object_label_source
object_label_sha256
random_seed
python_version
torch_version
cuda_version
cudnn_version
gpu_names
visible_gpu_ids
world_size
batch_size
num_workers
trace_float_dtype
eval_mode
no_grad
model_entry_file
model_entry_sha256
visual_graph_file
visual_graph_sha256
setm_file
setm_sha256
~~~

仓库当前是 dirty worktree。不要清理、reset 或覆盖用户已有改动。除 git commit 外，必须：

- 保存完整 git status；
- 对所有实际加载的新增/修改代码逐文件计算 SHA256；
- 将实际使用的关键代码复制到 provenance/code_snapshot/；
- 记录 command line 和完整 argparse Namespace。

full_args.json 必须保存运行时解析后的完整参数，而不是只保存手工挑选字段。

显式校验并记录以下 effective config：

~~~text
graph_text_topk = 5
graph_evidence_topk = 4
graph_num_layers = 2
graph_text_sync_mode = layerwise
graph_text_dynamic_retrieval = false
graph_enable_candidate_evidence = true
graph_enable_sync_evidence_matching = true
graph_enable_layerwise_sync_evidence_matching = true
graph_layerwise_sync_write_visual = true
graph_layerwise_sync_write_text = true
graph_layerwise_sync_final_matching = false
graph_layerwise_sync_disable_old_dense_injection = true
graph_store_dense_debug_tensors = true
num_hoi_queries = 32
~~~

如果 checkpoint profile、命令行和 parser default 不同，输出三方差异：

~~~text
checkpoint_value
command_line_value
parser_default
effective_value
resolution_reason
~~~

不得为满足提示词而擅自覆盖 checkpoint 实际 M/K 或结构。

### 6.1 门值

从实际加载 checkpoint 后的模型中保存 raw parameter 和 tanh 后的实际值：

~~~text
graph_gate_raw
graph_gate_tanh
candidate_evidence_gate_raw
candidate_evidence_gate_tanh
setm_visual_gate_raw
setm_visual_gate_tanh
setm_text_gate_raw
setm_text_gate_tanh
setm_anchor_gate_raw
setm_anchor_gate_tanh
~~~

同时保存 parameter name、shape、dtype、checkpoint key 和 load status。

### 6.2 Checkpoint 加载报告

生成 provenance/checkpoint_load_report.json：

~~~text
checkpoint_path
checkpoint_sha256
epoch
missing_keys
unexpected_keys
shape_mismatch_keys
loaded_parameter_count
loaded_parameter_bytes
strict_mode
M/K/profile_check
~~~

不得只依赖终端中的“load model weights”文本。

---

## 7. Phase 1：Prediction residual trace

真实正常路径大致为：

~~~python
base_hoi_query = rel_query_pre_proj(sampled_rel_reps)
graph_pair_features = graph_out.pair_feats
gated_query_delta = tanh(graph_gate) * graph_pair_features
corrected_hoi_query = base_hoi_query + gated_query_delta
decoder_hidden = interaction_decoder(corrected_hoi_query, ...)
final_action_logits = action_embed(decoder_hidden)
~~~

在 trace flag 开启且非训练状态时保存：

~~~text
base_hoi_query
graph_pair_features_final
graph_gate_raw
graph_gate_tanh
gated_query_delta
corrected_hoi_query
decoder_hidden_final
normal_final_action_logits
normal_final_action_probabilities
~~~

所有 tensor 保存完整 float32 到 PT/NPZ，不得只保存 norm。

每个 tensor 必须能回连：

~~~text
run_id
variant
image_id
file_name
candidate_id
mechanism_rank
original_pair_idx
selected_pair_idx
pair_idx
pair_valid
proposal_fingerprint
shape
dtype
~~~

生成 tables/query_residual_summary.csv，至少包含：

~~~text
base_query_l2
graph_pair_features_l2
query_delta_l2
corrected_query_l2
cosine_base_vs_graph_pair
cosine_base_vs_delta
cosine_base_vs_corrected
graph_gate_raw
graph_gate_tanh
residual_max_abs_error
~~~

必须验证：

~~~text
corrected_hoi_query
= base_hoi_query + tanh(graph_gate_raw) * graph_pair_features_final
~~~

有效 pair 的最大绝对误差：

~~~text
<= 1e-5
~~~

---

## 8. Phase 2：逐层实际写回状态与 pair feature

当前 visual_scene_graph_oov8.py 的旧 trace 在 visual/text write-back 之前构造。新 trace 必须同时区分：

- SETM 提议的 delta/update；
- 本 variant 是否实际应用；
- 实际写回后的状态。

对每一层，严格按真实执行顺序记录：

1. 当前视觉图层输出 updated_node_feats、updated_edge_attr；
2. 从当前 edge state gather refined_edge_pre_sync；
3. text_state.step；
4. 导出该层 prototype slots；
5. VET 构建 7 个 visual tokens；
6. SETM 计算 match_logits、双向概率、visual/text update 和 anchor delta；
7. 保存 proposed updates；
8. 按 variant 应用或跳过 visual write-back；
9. 按 variant 应用或跳过 text write-back；
10. 从实际 updated_edge_attr 重新 gather relation edge；
11. 使用实际 candidate-local text state 保存 post-writeback prototype slots；
12. 使用与最终输出相同的输入拼接和同一个 pair_out_mlp 构造该层 pair feature；
13. 只记录，不对正常 forward 增加第二次更新。

每层保存：

~~~text
layer_idx
pair_anchor_pre_sync
sync_anchor_delta_proposed
sync_anchor_delta_applied
pair_anchor_post_sync_actual
candidate_anchor_delta_proposed
candidate_anchor_delta_applied
graph_pair_features_after_layer
current_human_node_state
current_object_node_state
candidate_evidence_tokens
candidate_evidence_mask
candidate_evidence_weights
selected_context_indices
selected_context_mask
selected_hoi_indices
selected_hoi_mask
prototype_slots_pre_sync
prototype_slot_mask
updated_visual_tokens_proposed
updated_prototype_slots_proposed
prototype_slots_post_writeback_actual
match_logits
visual_to_text_probability
text_to_visual_probability
alpha
pi
role_coverage
visual_writeback_enabled
visual_writeback_applied
text_writeback_enabled
text_writeback_applied
setm_called
setm_correction_count
~~~

构造 graph_pair_features_after_layer 时必须使用：

~~~text
current_refined_edge_after_actual_writeback
current_human_node_state
current_object_node_state
base_pair_feats
union_input
~~~

并调用同一个已训练 pair_out_mlp。不得复制一个近似 MLP，也不得使用写回前 refined_edge_l。

---

## 9. Phase 3：原 decoder 的逐层只读 prediction probe

baseline full 对每个 candidate 构造 4 个 stage：

~~~text
base_query_decoder
after_layer_0_decoder
after_layer_1_decoder
normal_final_decoder
~~~

query 定义：

~~~text
base_query_decoder:
    base_hoi_query

after_layer_l_decoder:
    base_hoi_query
    + tanh(graph_gate) * graph_pair_features_after_layer_l

normal_final_decoder:
    正常 forward 已经计算的 corrected_hoi_query
~~~

要求：

- 调用同一个 interaction_decoder；
- 调用同一个 action_embed；
- 不把 coarse_action_probability 当成 layer-0；
- 不用线性近似；
- 不绕过 decoder 直接 action_embed(query)；
- normal final 使用正常 forward 的原始结果，不重复伪造；
- after_layer_1 必须由独立重建的 layer-1 pair feature 和原 decoder 得到，用于一致性验证。

当前 relation dependency encoding 是 content-aware 的，并依赖 query_reps。每个 probe query 都必须重新计算 dependency encoding。其他 pair topology、layout masks、decoder memory、position、memory padding mask 和 proposal order保持相同。

保存：

~~~text
probe_query
probe_dependency_encoding
probe_decoder_hidden_final
probe_action_logits_117
probe_action_probabilities_117
probe_top1_action
probe_strongest_competitor
~~~

必须验证：

~~~text
max_abs_diff(
  after_layer_1_decoder logits,
  normal_final_decoder logits
) <= 1e-5
~~~

通过前不得把逐层 prediction 数据标记为正式结果。

生成 tables/layer_prediction_scores.csv。

baseline 目标 20 pairs 的预期行数：

~~~text
20 × 4 stages × 117 actions = 9,360
~~~

每行字段至少包含：

~~~text
run_id
variant
seed
mechanism_rank
candidate_id
image_id
file_name
original_pair_idx
selected_pair_idx
pair_idx
proposal_fingerprint
stage
layer_idx
action_id_model
action_category_id_raw
action_name
action_logit
action_probability
is_gt_action
is_normal_final_top1
top1_action_id
top1_action_name
top1_probability
strongest_competitor_action_id
strongest_competitor_action_name
strongest_competitor_probability
~~~

---

## 10. Phase 4：细粒度 VET/SETM 更新表

生成：

~~~text
tables/visual_unit_updates.csv
tables/prototype_role_updates.csv
tables/anchor_writeback.csv
tables/context_transitions.csv
~~~

### 10.1 visual_unit_updates.csv

每行：

~~~text
target pair × layer × prototype × visual unit
~~~

包含所有 masked slot；invalid slot 使用 valid=false，并将不适用的数值写为空，不得伪装成普通零。

字段：

~~~text
run_id
variant
mechanism_rank
candidate_id
selected_pair_idx
layer_idx
prototype_idx
prototype_action_id
prototype_action_name
prototype_valid
visual_idx
visual_role
visual_valid
pre_norm_l2
post_norm_l2
delta_l2
delta_rms
relative_delta
cosine_pre_post
setm_alpha_weight
~~~

full baseline 形状完整行数：

~~~text
20 × 2 × 5 × 7 = 1,400
~~~

### 10.2 prototype_role_updates.csv

每行：

~~~text
target pair × layer × prototype × role
~~~

字段：

~~~text
run_id
variant
mechanism_rank
candidate_id
selected_pair_idx
layer_idx
prototype_idx
prototype_action_id
prototype_action_name
prototype_valid
text_bank_index
text_role_idx
text_role
text_role_raw
text_role_valid
pre_norm_l2
post_norm_l2
delta_l2
delta_rms
relative_delta
cosine_pre_post
prototype_role_coverage
setm_pi_weight
~~~

full baseline 形状完整行数：

~~~text
20 × 2 × 5 × 6 = 1,200
~~~

### 10.3 anchor_writeback.csv

字段：

~~~text
run_id
variant
mechanism_rank
candidate_id
selected_pair_idx
layer_idx
pair_valid
anchor_pre_l2
anchor_delta_proposed_l2
anchor_delta_applied_l2
anchor_post_actual_l2
anchor_relative_delta
cosine_pre_post
anchor_gate_raw
anchor_gate_tanh
candidate_anchor_delta_proposed_l2
candidate_anchor_delta_applied_l2
setm_anchor_delta_proposed_l2
setm_anchor_delta_applied_l2
visual_writeback_applied
~~~

full baseline 预期 40 行。

标准 full 配置应验证：

~~~text
candidate_anchor_delta_applied_l2 == 0
setm_anchor_delta_applied_l2 > 0 for effective valid pairs
~~~

不要混淆“模块计算出的 proposed delta”和“真实写入模型状态的 applied delta”。

### 10.4 context_transitions.csv

每个目标 pair 的 4 个 context slot 一行：

~~~text
run_id
mechanism_rank
candidate_id
selected_pair_idx
slot_idx
layer_0_node_idx
layer_0_node_name
layer_1_node_idx
layer_1_node_name
changed
layer_0_candidate_weight
layer_1_candidate_weight
layer_0_detector_score
layer_1_detector_score
layer_0_hk_valid
layer_0_ko_valid
layer_1_hk_valid
layer_1_ko_valid
~~~

预期：

~~~text
20 × 4 = 80 rows
~~~

---

## 11. Phase 5：官方 GT 与双重 pair mapping

不得只使用旧 trace 中可能为空的 gt_action_targets。

从官方 test_hico.json 重新导出 GT。必须区分两个映射：

### 11.1 原始 Top-20 到本轮 proposal

保存：

~~~text
original_pair_idx
selected_pair_idx
original_human_box
original_object_box
selected_human_box
selected_object_box
human_iou
object_iou
min_pair_iou
pair_mapping_warning
~~~

warning 阈值沿用上一轮：

~~~text
min_pair_iou < 0.8
~~~

不得直接假设 original_pair_idx 与 selected_pair_idx 相同。

已知上一轮：

- rank 6：object IoU 约 0.63，必须保留 warning；
- rank 19：human IoU 约 0.318，必须保留 warning，并默认排除出第五章成功案例。

### 11.2 本轮 predicted pair 到官方 GT pair

将 predicted normalized xyxy box 用原图尺寸转换到像素坐标后，与同一 image 的 GT subject/object boxes 分别计算 IoU。

默认有效条件：

~~~text
human_iou >= 0.5
object_iou >= 0.5
subject is person
object category compatible
~~~

同时：

- min_pair_iou < 0.8 时保留 mapping_warning；
- 一个 predicted pair 匹配多个 GT interaction 时，每个 GT interaction 一行；
- 多个 action 共享同一 GT pair 时，不得只保留一个；
- 没有可靠 GT 时保留该 candidate，并写出明确原因；
- 禁止用模型预测补成 GT。

生成 tables/gt_pair_mapping.csv：

~~~text
run_id
mechanism_rank
candidate_id
image_id
file_name
original_pair_idx
selected_pair_idx
pred_human_box_xyxy_norm
pred_object_box_xyxy_norm
pred_human_box_xyxy_pixel
pred_object_box_xyxy_pixel
gt_human_box_xyxy_pixel
gt_object_box_xyxy_pixel
human_iou
object_iou
min_pair_iou
gt_hoi_id_raw_1based
gt_hoi_id_zero_based
gt_action_category_id_raw
gt_action_id_model
gt_action_name
gt_object_category_id_raw
gt_object_id_model
gt_object_name
gt_available
mapping_valid
mapping_warning
mapping_method
failure_reason
~~~

本提示词中 action_id_model 始终指模型 117 类 logits 的 0-based index；官方 annotation 的 category_id 另存为 action_category_id_raw。不要混用。

重点核对旧 trace GT action 为空的 ranks：

~~~text
1, 8, 11, 13, 14, 19, 20
~~~

逐个标记：

~~~text
no_compatible_gt_pair
gt_pair_found_but_old_export_missing
proposal_pair_remapping_error
raw_to_model_action_mapping_error
low_human_iou
low_object_iou
object_category_mismatch
~~~

---

## 12. Phase 6：5 个确定性机制 variant

建议新增一个仅用于 evaluation trace 的统一参数：

~~~text
--graph_mechanism_validation_variant
~~~

choices：

~~~text
full
no_anchor_writeback
no_prototype_state_writeback
no_setm
endpoint_only_setm
~~~

所有 variant：

- 同一 checkpoint；
- 同一 20 张图；
- 同一 seed；
- 同一 world size；
- 同一 batch size；
- 同一 detector proposals；
- 同一 pair order；
- 同一 32 candidates；
- 同一 Top-5 retrieval；
- eval/no_grad；
- 不训练；
- 不更新 prototype bank；
- 输出完整 117 类 logits/probabilities。

每张图生成 proposal_fingerprint，至少由以下内容 hash 得到：

~~~text
pair_indices
human/object boxes
detector labels
proposal scores
pair order
~~~

每个 pair 生成 retrieval_fingerprint，至少由以下内容 hash 得到：

~~~text
selected_hoi_indices
selected_hoi_mask
selected prototype action/object IDs
~~~

variant 输出只有在 fingerprint 与 full 一致时才有效。任何不一致必须中止该 variant 的正式汇总并定位原因。

固定 random、NumPy、PyTorch 和 CUDA seeds。记录 deterministic algorithm 设置。不得依赖 raw pair_idx 跨运行自然相同。

### 12.1 full

~~~text
write_visual = true
write_text = true
SETM called = true
context mask = normal 7 units
~~~

### 12.2 no_anchor_writeback

SETM 仍计算：

- match_logits；
- visual/text 双向概率；
- updated visual tokens；
- updated prototype slots；
- alpha/pi/coverage；
- proposed anchor delta。

但：

~~~text
write_visual = false
write_text = true
sync_anchor_delta_applied = 0
~~~

不得把 proposed anchor delta 误记成 applied write-back。

### 12.3 no_prototype_state_writeback

~~~text
write_visual = true
write_text = false
~~~

仍保存 proposed updated_prototype_slots，但下一层使用的 actual candidate-local prototype state 不得被该 update 改写。

### 12.4 no_setm

保留相同 backbone、proposals、基础视觉图、VET context selection 和只读 prototype retrieval，但：

- 不调用 SETM matcher；
- 不生成或应用 SETM visual/text/anchor update；
- 不自动回退到 post-hoc/final SETM；
- 不启用 old dense injection；
- 不应用 candidate-only final anchor delta 来替代 SETM；
- setm_called=false；
- setm_correction_count=0。

当前代码只关闭 layerwise flag 可能触发旧 final candidate/SETM 分支，因此不得仅靠组合旧 flags 猜测 no_setm。应实现明确的 evaluation-only variant 分支并验证实际调用次数。

### 12.5 endpoint_only_setm

VET 正常构建 7 个 tokens，context selection 不变。仅在送入 SETM 前，将：

~~~text
P_1, P_2, P_3, P_4
~~~

对应的 visual mask 设为 false，只保留：

~~~text
E_ho, X_h, X_o
~~~

不得删除节点、重建 proposals、改变 pair order 或改变 retrieval。保存：

~~~text
original_visual_mask
intervened_visual_mask
masked_visual_indices = [3, 4, 5, 6]
~~~

### 12.6 默认不运行的可选实验

以下不是本轮必需交付，默认跳过：

~~~text
pair_mismatched_context
object_compatible_mismatched_prototypes
~~~

在 README 中写明“not requested in deterministic minimum scope”，不要伪造 skipped donor 数据。

---

## 13. Variant 输出与因果对照表

生成 tables/ablation_scores.csv。

确定性最小范围且无跳过时预期行数：

~~~text
5 variants × 20 pairs × 117 actions = 11,700
~~~

字段：

~~~text
run_id
variant
seed
mechanism_rank
candidate_id
image_id
file_name
original_pair_idx
selected_pair_idx
pair_idx
proposal_fingerprint
retrieval_fingerprint
action_id_model
action_category_id_raw
action_name
action_logit
action_probability
full_action_logit
full_action_probability
action_logit_delta_vs_full
action_probability_delta_vs_full
is_gt_action
is_baseline_top1
baseline_top1_action_id
variant_top1_action_id
variant_top1_action_name
prediction_changed
analysis_target_action_id
analysis_target_source
target_probability_delta_vs_full
strongest_competitor_action_id
competitor_probability_delta_vs_full
setm_called
visual_writeback_applied
text_writeback_applied
skipped
skipped_reason
~~~

对于每一行，action_probability_delta_vs_full 定义为该 action 在 variant 与 full 的直接差值。

analysis_target_action 的确定规则：

1. 若 GT mapping 有效，优先记录所有 GT actions，并在单独 summary 中分别统计；
2. 若需要单一目标用于案例排序，使用 candidate 原始机制排名所对应的 action；
3. 若该 action 不属于可靠 GT，不得称为 correct target；
4. 不允许用 variant 的新 top-1 动态改变分析目标。

另外生成 tables/ablation_case_summary.csv，每个 variant × candidate 一行，用于第五章直接筛选。

---

## 14. 统一自包含数据包

最终目录至少为：

~~~text
evidence_trace_top20_mechanism_validation_v2_<timestamp>/
├── README.md
├── run_meta.json
├── config/
│   ├── full_args.json
│   ├── environment.json
│   ├── file_hashes.json
│   ├── variant_config_diff.json
│   └── variants/
│       └── <variant>_effective_args.json
├── manifests/
│   ├── top20_cases.csv
│   ├── source_asset_manifest.csv
│   ├── proposal_fingerprints.csv
│   └── retrieval_fingerprints.csv
├── tables/
│   ├── layer_prediction_scores.csv
│   ├── query_residual_summary.csv
│   ├── visual_unit_updates.csv
│   ├── prototype_role_updates.csv
│   ├── anchor_writeback.csv
│   ├── context_transitions.csv
│   ├── gt_pair_mapping.csv
│   ├── ablation_scores.csv
│   └── ablation_case_summary.csv
├── arrays/
│   └── baseline full 每个目标案例一个 NPZ
├── raw_records/
│   ├── full/
│   ├── no_anchor_writeback/
│   ├── no_prototype_state_writeback/
│   ├── no_setm/
│   └── endpoint_only_setm/
├── source_images/
│   └── 固定 Top-20 JPEG
├── labels/
│   ├── action_names.json
│   ├── object_names.json
│   ├── action_id_mapping.json
│   ├── object_id_mapping.json
│   └── prototype_prompt_metadata.json
├── provenance/
│   ├── original_to_validation_pair_mapping.csv
│   ├── checkpoint_load_report.json
│   ├── baseline_equivalence_report.json
│   ├── prototype_bank_integrity.json
│   ├── validation_report.json
│   ├── commands.txt
│   └── code_snapshot/
├── reports/
│   ├── code_audit.md
│   ├── field_dictionary.md
│   ├── result_summary.md
│   └── warnings.md
└── logs/
    ├── baseline.log
    └── variants/
        └── <variant>.log
~~~

每个 NPZ 至少包含：

~~~text
base_hoi_query
graph_pair_features_after_layer_0
graph_pair_features_after_layer_1
graph_pair_features_final
gated_query_delta
corrected_hoi_query
probe_queries
probe_decoder_hidden
probe_action_logits
probe_action_probabilities
pair_anchor_pre_sync
sync_anchor_delta_proposed
sync_anchor_delta_applied
pair_anchor_post_sync_actual
candidate_evidence_tokens
candidate_evidence_mask
candidate_evidence_weights
prototype_slots_pre_sync
updated_prototype_slots_proposed
prototype_slots_post_writeback_actual
prototype_slot_mask
match_logits
visual_to_text_probability
text_to_visual_probability
alpha
pi
role_coverage
selected_context_indices
selected_hoi_indices
~~~

所有 CSV：

- UTF-8；
- 第一行字段名；
- 浮点保留至少 8 位有效数字；
- boolean 写 true/false；
- list/tensor 不塞入不可解析的 Python repr，使用 JSON；
- invalid/masked slot 的不适用浮点写空；
- 不得把 invalid 值伪装成 0；
- 所有行至少可通过 run_id + variant + candidate_id + pair_idx + layer_idx 回连；
- schema 和字段语义写入 reports/field_dictionary.md。

---

## 15. 必须通过的验证

扩展或新增验证脚本，生成 provenance/validation_report.json。每个检查包含：

~~~text
name
status
expected
actual
tolerance
affected_candidates
message
~~~

至少检查：

1. 固定 20 张图片全部存在；
2. Top-20 candidate_id/image_id 与原清单完全一致；
3. 每张图正式 trace 有 32 pairs；
4. 每个目标 pair 映射完成或有明确 warning；
5. full 每个目标 pair 有两层 trace；
6. M=4、visual units=7、K=5、roles=6、layers=2；
7. 所有有效 float tensor 无 NaN/Inf；
8. 所有 masks 与 tensor shape 一致；
9. Top-5 prototype IDs 在 full 两层之间不变；
10. 非 prototype intervention variants 与 full retrieval_fingerprint 一致；
11. 所有 variants 与 full proposal_fingerprint 一致；
12. Frozen Prototype Bank 的 before/after SHA256 完全一致；
13. 每层只有一次 SETM anchor correction；
14. layerwise loop 后没有额外 final SETM；
15. candidate-only anchor delta 没有成为第二次主要写回；
16. corrected_query = base_query + gated_query_delta；
17. residual max_abs_error <= 1e-5；
18. after_layer_1_decoder 与 normal_final logits max_abs_diff <= 1e-5；
19. integer/mask/index tensor 完全一致；
20. normal final top-1 与 reference baseline 一致；
21. full 新 trace 与旧详细 Top-20 reference 按 boxes 映射后等价；
22. no_anchor_writeback proposed delta 可存在，但 applied delta 必须为 0；
23. no_prototype_state_writeback 的 actual text state 不得写入 proposed update；
24. no_setm 的 SETM 调用数必须为 0；
25. no_setm 不得落入 final/post-hoc SETM 或 old dense injection；
26. endpoint_only_setm 只 mask visual indices 3/4/5/6；
27. GT 类别没有进入 retrieval；
28. ranks 6/19 warning 未被删除；
29. valid prototype prompt 文本完整；
30. layer_prediction_scores.csv 为 9,360 行；
31. deterministic ablation_scores.csv 为 11,700 行；
32. 所有表唯一键、行数、缺失值规则通过；
33. 每个 variant 日志与 effective args 存在；
34. checkpoint、代码、annotation、prototype 与 Top-20 均有 hash。

### 15.1 Baseline equivalence

旧详细 Top-20 trace 是本轮 pre-modification reference：

~~~text
/wzj/zijian/OSSM-HOI-main/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed/traces
~~~

比较时：

1. 先按 image_id 匹配；
2. 再按 human/object boxes 最大 pair IoU 匹配；
3. proposal_fingerprint 完全一致时才执行严格 tensor diff；
4. 不得直接用 raw pair_idx 对齐；
5. 如果 proposal fingerprint 不一致，先修复确定性设置并重跑，不能将不匹配的 pair 当成数值误差。

严格容差：

~~~text
float tensor max_abs_diff <= 1e-5
integer/mask/index exact match
top-1 action ID exact match
~~~

新增 probe 必须在 normal final 计算之后或只读分支中运行，不得影响 normal final。

---

## 16. 建议新增或扩展的脚本

优先扩展已有工具，不要重复造多个功能重叠脚本。建议最终提供：

~~~text
run_top20_mechanism_validation_v2.sh
tools/evidence_vis/export_mechanism_validation_v2.py
tools/evidence_vis/export_official_gt_pair_mapping.py
tools/evidence_vis/validate_mechanism_validation_v2.py
tools/evidence_vis/compare_mechanism_variants.py
~~~

运行 wrapper 必须：

- 将 checkpoint、DETR、BLIP、M、K、GPU_IDS、batch size 写在顶部用户修改区；
- 默认使用本提示词第 1 节的真实路径；
- GPU_IDS 可由环境变量覆盖；
- 每个 variant 使用相同 GPU_IDS/world size/batch size；
- 生成唯一 RUN_TAG；
- 输出完整命令；
- 任一正式验收失败时退出非零；
- 保留已完成 variant，不删除历史输出；
- 不覆盖旧目录。

建议默认值：

~~~text
RESUME_CHECKPOINT=/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth
DETR_WEIGHTS=/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth
BLIP_PRETRAINED=/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth
BLIP_VIT=large
BACKBONE=resnet101
GRAPH_EVIDENCE_TOPK=4
GRAPH_TEXT_TOPK=5
BATCH_SIZE=1
NUM_WORKERS=2
TRACE_FLOAT_DTYPE=float32
~~~

GPU_IDS 不应假设服务器永远使用同一张卡。运行前检查 GPU 状态；若用户已通过环境变量指定，则严格使用用户指定值。

---

## 17. 实施顺序

按以下顺序执行，不要一开始就启动正式 GPU 推理：

1. 资产和路径审计；
2. import 链和算法路径审计；
3. 冻结 pre-run metadata/hash；
4. 实现 trace 与 variant flags；
5. 静态语法检查；
6. 运行相关 CPU/unit/smoke tests；
7. 用 Top-20 第一张图做一次 smoke；
8. 验证 residual 和 layer-1 probe 一致性；
9. 正式运行 full Top-20；
10. 与旧详细 reference 做 baseline equivalence；
11. 只有 full 通过后运行 4 个 deterministic variants；
12. 校验 proposal/retrieval fingerprints；
13. 离线导出官方 GT mapping；
14. 生成 CSV/NPZ/PT 和说明文档；
15. 运行统一 validation；
16. 生成摘要；
17. 打包和计算 SHA256。

如 smoke 或 full baseline 不通过，不得继续生成正式 ablation 表。

---

## 18. 最终打包

验证全部通过后，将最终目录打包为：

~~~text
evidence_trace_top20_mechanism_validation_v2_<timestamp>.tar.zst
~~~

压缩后执行：

- zstd 完整性检查；
- tar 成员读取检查；
- 源文件数与归档文件数对比；
- 关键文件存在性检查；
- SHA256。

最终报告：

~~~text
archive_path
archive_size_bytes
archive_sha256
source_file_count
archive_file_count
integrity_passed
~~~

---

## 19. 最终回复必须包含

用简短、可核对的中文总结：

1. 实际使用的 checkpoint 路径和 SHA256；
2. 实际模型入口；
3. git commit 和 dirty 状态；
4. 最终 effective M/K/layers；
5. full 是否 20/20 完整；
6. 可靠 GT mapping 数量；
7. ranks 6/19 的最终状态；
8. 每个 deterministic variant 成功/失败/跳过数量；
9. baseline equivalence 是否通过；
10. residual identity 最大误差；
11. layer-1 probe 与 normal final 最大误差；
12. proposal/retrieval fingerprint 是否一致；
13. prototype bank before/after hash 是否一致；
14. 所有 warning 和未解决问题；
15. 实际修改的代码文件；
16. 新增脚本；
17. 数据包目录；
18. tar.zst 路径、大小和 SHA256。

如果某项无法完成，保留已完成产物，并精确写明：

~~~text
failed_phase
missing_file
failed_command
exit_code
error_message
attempted_fixes
verified_fallback
required_user_action
~~~

不要用“可能”“大概”替代已经可以从服务器检查的事实。

---

## 20. 明确禁止

- 禁止重跑完整 HICO-DET 测试集；
- 禁止训练或更新参数；
- 禁止更新 prototype bank；
- 禁止用 GT 类别参与 retrieval；
- 禁止重新挑选 Top-20；
- 禁止假设跨运行 pair_idx 不变；
- 禁止把 coarse probability 当逐层 decoder probability；
- 禁止绕过原 decoder；
- 禁止把 proposed update 当 actual write-back；
- 禁止在 layerwise SETM 后再做 final SETM；
- 禁止用 old dense injection 替代 no_setm；
- 禁止把预测 action 补成 GT；
- 禁止静默删除 rank 6/19 warning；
- 禁止覆盖旧 trace、旧 package、旧图或 checkpoint；
- 禁止清理 dirty worktree；
- 禁止只输出计划而不实际执行；
- 禁止在验收失败时仍将数据标记为正式可用。

---

## 21. 开始执行

从仓库和资产审计开始。先输出简短的审计结论与预期修改文件，然后继续实施，不要等待用户重复确认。

仅在 checkpoint、Top-20 清单、数据标注或必要权重确实缺失，或者 GPU 使用需要新的用户授权时暂停。

完成代码、smoke、正式 Top-20、5 个 deterministic variants、GT mapping、统一验证和打包后，再提交最终结果。
