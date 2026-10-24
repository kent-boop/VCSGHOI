# OSSM-BOX：BLIP ROI 无 SAM 版本

## 当前结构

`ossm_box` 是独立模型变体。它保留旧 OSSM 的分层文本选择、动态适配、
query/memory 增强、关系解码器、损失和后处理，不构建 SAM，也不叠加 Evidence
图推理模块。

三专家 MoE 的固定语义为：

1. `global`：与旧模型一致，使用原图 BLIP 最后一层 CLS，并按实际关系数展开。
2. `instance`：在增强图像的 BLIP 倒数第 2 层 patch map 上分别 ROIAlign 人框和
   物框，再做有方向的人物对融合。
3. `action`：在同一增强图像的 BLIP 最后一层 patch map 上 ROIAlign 人物外接
   union box。

BLIP `multilscale` 的底层实现返回倒数第 3、倒数第 2、倒数第 1 层。旧主线仍按
`scene patches / instance patches / action CLS` 使用这三层；ROI 分支复用其中的
instance 和 action 层，不改变 scene/instance memory 路径。

需要区分：旧模型的 SAM/union 区域图各自通过 `itc` 单独编码，二者实际都取最后
一层 CLS；“最后三层”原本描述的是全局分层主线。现在按已确认的新 expert 语义，
instance ROI 明确使用倒数第 2 层空间 patch，action ROI 使用最后一层空间 patch。

## ROI 与人物对融合

ROI 图像来自 DETR 当前看到的增强样本：先去掉 `NestedTensor` 的 batch padding，
反解 ImageNet 归一化，再 stretch 到 BLIP 的 384×384 输入。因此 DETR 的 normalized
`cxcywh` 框只需转换并裁剪成 normalized `xyxy`，在 24×24 patch map 上按网格尺寸
缩放即可。

ROIAlign 使用 `output_size=5`、`sampling_ratio=2`、`aligned=True`。instance 与
action 使用各自的 `LayerNorm + Linear + ReLU + Linear` ROI 投影器。

人物对是有方向的。设归一化后的 ROI 特征为 `h` 和 `o`，instance 投影器输入为：

```text
[h, o, h * o, h - o]  ->  LayerNorm  ->  Linear(1024,512)
                       ->  GELU       ->  Linear(512,256) -> L2 normalize
```

交换人和物会同时交换前两段并翻转差分项，因而不会退化成对称平均。

## 资产与训练

正式脚本为 `run_ossm_box_hico.sh`。它不加载旧 OSSM checkpoint；新的 ROI
投影器、人物对融合、MoE、adapter 和 HOI head 从头训练。BLIP-L 与 DETR 仍使用
预训练权重，这里的 scratch 指“不用旧 HOI/OSSM 权重”，不是将两个基础视觉
编码器随机初始化。

默认资产：

- BLIP-L：旧项目的 `model_large_retrieval_coco.pth`。
- 分层文本库：旧项目原文件 `hico_L_embeds_256.npy`，shape 为 `520×3×256`。
- DETR：HICO ResNet-101 `checkpoint0105.pth`。
- 对象动作先验：`obj_act_prior.npy`；当前工作区文件与旧项目文件 SHA-256 相同。

`--resume` 仍可用于将来续训同结构的 OSSM-ROI checkpoint，但正式 scratch 脚本
不会传入 `--init_from_ossm` 或旧 OSSM checkpoint。

## 验证

轻量测试覆盖 normalized 框转换、union box、ROI 数量/空输入、有序人物融合、
增强图像 padding、训练 hard-mining、随机采样、推理、完整 decoder forward/backward、
新 ROI 模块梯度、冻结 BLIP/DETR 以及 SAM/Evidence 依赖隔离。

真实单批次验证可运行：

```bash
python tools/smoke_ossm_box.py \
  --detr_checkpoint /path/to/resnet101_detr.pth \
  --blip_checkpoint /path/to/model_large_retrieval_coco.pth \
  --text_library /path/to/hico_L_embeds_256.npy \
  --prior obj_act_prior.npy \
  --data_path /path/to/hico_20160224_det \
  --device cuda:0
```

它只加载预训练 DETR/BLIP，不加载旧 OSSM checkpoint。

本机真实 HICO 单批次已完成 forward、criterion、backward、optimizer step、eval 和
postprocess：train/eval 均为 32 个关系，verb 输出 `[32,117]`，223 个可训练张量有
有限梯度，冻结 DETR/BLIP 的可训练参数均为 0，`old_ossm_initialized=false`。

## 主要文件

- `src/models/stip_ossm_box.py`：OSSM 主路径和三种 expert 输入。
- `src/models/ossm_blip_roi.py`：增强图像预处理、框转换和 ROIAlign。
- `src/models/moe_fusion_box.py`：`global/instance/action` 三专家 MoE。
- `src/models/ossm_relation_features.py`：关系特征及对象动作先验。
- `run_ossm_box_hico.sh`：无旧 OSSM 初始化的 HICO 训练入口。
- `tools/smoke_ossm_box.py`：真实单批次验收。
