import torch

# 1. 加载你的 sichoi 模型路径
ckpt_path = "/share/share/ljg/SICHOI/checkpoints/vcoco/KakaoBrain/run_000001/best_large.pth"
checkpoint = torch.load(ckpt_path, map_location='cpu')

# 2. 获取模型状态字典
state_dict = checkpoint['model']

# 3. 检查是否存在 backbone 权重
has_backbone = any('backbone' in k for k in state_dict.keys())
print(f"包含 Backbone 权重: {has_backbone}")

# 4. 打印一些层的名字确认一下 (例如 layer4 是 ResNet 的高层)
resnet_keys = [k for k in state_dict.keys() if 'backbone.0.body.layer4' in k]
print(f"检测到 ResNet 层数量: {len(resnet_keys)}")

# 5. 如果你想把这个当做纯粹的 DETR/ResNet 预训练权重加载
# 通常我们只需要 'detr.' 开头的权重（如果你的新模型结构也是 STIP 套 DETR）
# 或者你可以清洗一下 key，去掉 'detr.' 前缀或者保留，取决于你 STIP_main.py 里加载的逻辑