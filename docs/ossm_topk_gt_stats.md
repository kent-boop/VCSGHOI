# BLIP CLS Top-k GT 文本检索统计

运行（脚本默认加载当前 run 的 best.pth）：

```bash
bash run_hicodet_ossm_roi_r101_blipl.sh --validation --ossm_topk_gt_stats
```

删除 `--ossm_topk_gt_stats` 即关闭。`--validation` 是 `--eval` 的别名；当前脚本本身已带 `--eval`。本实验只支持 HICO-DET 的 ossm_box 验证，不改变模型权重、Top-k 选择或检测结果。

结果位于运行输出目录下的 `topk_gt_stats/`，默认是：
`checkpoints/hico-det_roi_r101_blipl/STIP_opensource/hico_ossm_roi_r101_blipl_scratch/topk_gt_stats/`。
每次运行覆盖该目录内同名结果文件；需要保留多次实验时可设置不同的 `RUN_NAME`（并通过 `RESUME_CHECKPOINT` 指定权重）。

- `summary.json`：scene Top-3、instance Top-10、action Top-20 的统计及权重、文本库路径。
- `per_image.jsonl`：每张图一行，包含实际 Top-k 的文本库索引、文本，以及每条 GT 的 `in_library`、`hit`、`rank_in_topk`。排名从 1 开始，未命中为 null；类别和文本行索引从 0 开始。

推荐关注 `branches.action.image_class_recall`：先将每张图中的相同 GT HOI 类别去重，然后用命中的图像–类别数除以全部有效图像–类别数。`instance_recall` 按 GT 人物–物体–动作实例计数；`any_hit_image_rate` 和 `all_hit_image_rate` 分别表示至少命中一个、命中全部有效 GT 类别的图片比例。比例范围为 0–1；没有有效分母时输出 null。

文本行顺序使用生成代码中的 `hico_lib.py` 字典插入顺序，而非 600 类官方 HOI ID。GT 对象使用 COCO 稀疏 category_id 到 80 类连续索引的映射，与动作索引组成键。520 行文本库不包含 80 个 no_interaction 类：这些 GT 在逐图文件中保留并标记 `in_library=false`，在汇总中单独计数，不计入召回率分母。若替换为行顺序不同的自定义文本库，必须同步修改映射。

该实验衡量整张图 CLS 的文本检索覆盖率，不做预测框与 GT 框匹配，不等于 HOI 检测召回率或 mAP。正常 mAP 验证仍会执行。多卡采样造成的重复图像会按 image_id 去重；无有效 GT 的图片不会被计为“全部命中”。
